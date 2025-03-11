import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from utils.inc_net import IncrementalNet, SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet_adapt_AC, \
    SimpleVitNet
from utils.AC_net import IncrementalNet, SimpleCosineIncrementalNet, SimpleVitNet, SimpleVitNet_AL
from models.base import BaseLearner
from backbone.linears import CosineLinear
from utils.toolkit import target2onehot, tensor2numpy
import copy
from torch.utils.data import DataLoader, Dataset,TensorDataset,random_split

num_workers = 8

# 创建一个新的 全连接层
class SimpleNN(nn.Module):  
    def __init__(self, input_size, output_size,dtype=torch.float32):  
        super(SimpleNN, self).__init__()  
        # 初始化全连接层  
        # input_size: 输入特征的数量  
        # output_size: 输出特征的数量（即该层的神经元数量）  
        self.fc = nn.Linear(input_size, output_size)  
  
    def forward(self, x):  
        if x.dtype != self.fc.weight.dtype:  
            x = x.to(dtype=self.fc.weight.dtype)  
        x = self.fc(x)  
        return x  
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["backbone_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')

        if 'resnet' in args['backbone_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self.batch_size = 128
            self.init_lr = args["init_lr"] if args["init_lr"] is not None else 0.01
        else:
            # self._network = SimpleVitNet(args, True)
            self._network = SimpleVitNet_AL(args, True)
            self.batch_size = args["batch_size"]
            self.init_lr = args["init_lr"]
            self.progressive_lr = args["progressive_lr"]
        self.model_hidden = args["Hidden"] 
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.R = None
        self._means = []
        self._cov_matrix = []
        self._std_deviations_matrix = []

    def after_task(self):
        self._known_classes = self._total_classes
        # calculate before update the old_model

        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        if self._cur_task == 0:
            self._network.fc = CosineLinear(self._network.feature_dim, self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)

        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                      weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.construct_dual_branch_network()
            self._network.update_fc(self._total_classes, self.args['Hidden'])

        else:
            self._network.update_fc(self._total_classes, self.args['Hidden'], cosine_fc=True)
            self._network.update_fc(self._total_classes, self.args['Hidden'])
            for param in self._network.ac_model.parameters():
                param.requires_grad = False
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            # if total_params != total_trainable_params:
            #     for name, param in self._network.named_parameters():
            #         if param.requires_grad:
            #             print(name, param.numel())
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.progressive_lr,
                                      weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.progressive_lr,
                                        weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['progreesive_epoch'],
                                                             eta_min=self.min_lr)
            self._progreessive_train(train_loader, test_loader, optimizer, scheduler)

        if self._cur_task == 0:
            self._compute_means()
            self._network.to(self._device)
            # AL training process
            self.cls_align(train_loader, self._network)
        else:
            self._compute_means()
            self.cali_prototye_model(train_loader)
            self._compute_relations()
            self._build_feature_set()
            self._network.to(self._device)
            # AL training process
            self.IL_align(train_loader, self._network)
            self.cali_weight(self._feature_trainset, self._network)

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet_adapt_AC(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network = network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _progreessive_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['progreesive_epoch']))


        EMA_model = self._network.copy().freeze()
        alpha = self.args['alpha']

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["train_logits"]

                loss_ce = F.cross_entropy(logits, targets)
                loss = loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            for param, ema_param in zip(self._network.backbones[0].parameters(), EMA_model.backbones[0].parameters()):
                ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_ac_train_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['progreesive_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        for param, ema_param in zip(EMA_model.backbones[0].parameters(),
                                    self._network.backbones[0].parameters()):
            ema_param.data =  param.data

        logging.info(info)

    def cls_align(self, trainloader, model):
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        embedding_list = []
        label_list = []

        # AL training process
        model = model.eval()

        auto_cor = torch.zeros(model.ac_model.fc[-1].weight.size(1), model.ac_model.fc[-1].weight.size(1)).to(
            self._device)
        crs_cor = torch.zeros(model.ac_model.fc[-1].weight.size(1), self._total_classes).to(self._device)

        with torch.no_grad():
            pbar = tqdm(enumerate(trainloader), desc='Alignment', total=len(trainloader), unit='batch')
            for i, batch in pbar:
                (_, data, label) = batch
                images = data.to(self._device)
                target = label.to(self._device)

                label_list.append(target.cpu())

                feature = model(images)["features"]
                new_activation = model.ac_model.fc[:2](feature)

                embedding_list.append(new_activation.cpu())

                label_onehot = F.one_hot(target, self._total_classes).float()
                auto_cor += torch.t(new_activation) @ new_activation
                crs_cor += torch.t(new_activation) @ (label_onehot)

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        Y = target2onehot(label_list, self._total_classes)

        ridge = self.optimise_ridge_parameter(embedding_list, Y)
        logging.info("gamma {}".format(ridge))

        print('numpy inverse')
        R = np.mat(auto_cor.cpu().numpy() + ridge * np.eye(model.ac_model.fc[-1].weight.size(1))).I
        R = torch.tensor(R).float().to(self._device)
        Delta = R @ crs_cor
        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9 * Delta.float()))
        self.R = R
        del R

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda =',ridge)
        return ridge

    def IL_align(self, trainloader, model):
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        # AL training process
        model = model.eval()

        W = (model.ac_model.fc[-1].weight.t()).float()
        R = copy.deepcopy(self.R.float())

        with torch.no_grad():
            pbar = tqdm(enumerate(trainloader), desc='Alignment', total=len(trainloader), unit='batch')
            for i, batch in pbar:
                (_, data, label) = batch
                images = data.to(self._device)
                target = label.to(self._device)

                feature = model(images)["features"]
                new_activation = model.ac_model.fc[:2](feature)
                label_onehot = F.one_hot(target, self._total_classes).float()

                R = R - R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.size(0)).to(self._device) +
                    new_activation @ R @ new_activation.t()) @ new_activation @ R

                W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)

        print('numpy inverse')
        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        self.R = R
        del R


    def _compute_means(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                           source='train',
                                                                           mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self.extract_prototype(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)

                # 计算协方差矩阵
                cov = np.cov(vectors, rowvar=False)
                self._cov_matrix.append(cov)

                # 提取对角线元素（方差），即各个特征的方差
                variances = np.diagonal(cov)
                # 计算各个特征的标准差
                std_deviations = np.sqrt(variances)
                self._std_deviations_matrix.append(std_deviations)

    def _compute_relations(self):
        old_means = np.array(self._means[:self._known_classes])
        new_means = np.array(self._means[self._known_classes:])
        self._relations = np.argmax((old_means / np.linalg.norm(old_means, axis=1)[:, None]) @ (
                new_means / np.linalg.norm(new_means, axis=1)[:, None]).T, axis=1) + self._known_classes

    def extract_prototype(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(self._network(_inputs.to(self._device))["features"])
            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _build_feature_set(self):
        self.vectors_train = []
        self.labels_train = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
                                                                       source='train',
                                                                       mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
            vectors, _ = self.extract_prototype(idx_loader)
            self.vectors_train.append(vectors)
            self.labels_train.append([class_idx] * len(vectors))
        for class_idx in range(0, self._known_classes):
            new_idx = self._relations[class_idx]
            self.vectors_train.append(
                self.vectors_train[new_idx - self._known_classes] - self._means[new_idx] + self._means[class_idx])
            self.labels_train.append([class_idx] * len(self.vectors_train[-1]))

        self.vectors_train = np.concatenate(self.vectors_train)
        self.labels_train = np.concatenate(self.labels_train)
        self._feature_trainset = FeatureDataset(self.vectors_train, self.labels_train)
        self._feature_trainset = DataLoader(self._feature_trainset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)


    def cali_weight(self, cali_pseudo_feature, model):
        if hasattr(model, 'module'):
            model = model.module
        else:
            model = model

        # AL training process
        model = model.eval()

        W = (model.ac_model.fc[-1].weight.t()).float()
        R = copy.deepcopy(self.R.float())

        with torch.no_grad():
            pbar = tqdm(enumerate(cali_pseudo_feature), desc='Alignment', total=len(cali_pseudo_feature), unit='batch')
            for i, batch in pbar:
                (_, data, label) = batch
                features = data.to(self._device)
                target = label.to(self._device)

                new_activation = model.ac_model.fc[:2](features.float())
                label_onehot = F.one_hot(target, self._total_classes).float()

                # 获取wrong prediction的索引
                output = model.ac_model.fc[-1](new_activation)
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                false_indices = (correct == False).view(-1).nonzero(as_tuple=False)

                new_activation = new_activation[false_indices[:, 0]]
                label_onehot = label_onehot[false_indices[:, 0]]

                R = R - R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.size(0)).to(self._device) +
                    new_activation @ R @ new_activation.t()) @ new_activation @ R

                W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)

        print('numpy inverse')
        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        self.R = R
        del R

    def cls_align_calimodel(self,trainloader, in_features,hidden_size,out_dim):
        embedding_list = []
        label_list = []
        model =self._network.generate_fc(in_features,hidden_size,out_dim)
        model.fc[0].weight.data = self._network.ac_model.fc[0].weight.data
        assert torch.allclose(model.fc[0].weight.data, self._network.ac_model.fc[0].weight.data)
        model= model.to(self._device)
        #自相关矩阵  
        auto_cor = torch.zeros(model.fc[-1].weight.size(1), model.fc[-1].weight.size(1)).to(self._device)
        #交叉相关矩阵
        crs_cor = torch.zeros(model.fc[-1].weight.size(1), model.fc[-1].weight.size(0)).to(self._device)
        with torch.no_grad():
            pbar = tqdm(enumerate(trainloader), desc='cls_align_calimodel', total=len(trainloader), unit='batch')
            for i, batch in pbar:
                X_tensor,y_tensor = batch
                X_tensor = X_tensor.to(self._device)
                y_tensor = y_tensor.to(self._device)
                #print(y_tensor)
                label_list.append(y_tensor.cpu())
                new_activation= model.fc[:2](X_tensor)
                embedding_list.append(new_activation.cpu())
                auto_cor += torch.t(new_activation) @ new_activation 
                crs_cor += torch.t(new_activation) @ y_tensor
                
        embedding_list= torch.cat(embedding_list, dim=0)
        label_list= torch.cat(label_list, dim=0)
        ridge = self.optimise_ridge_parameter(embedding_list, label_list)
        R = np.mat(auto_cor.cpu().numpy() + ridge * np.eye(model.fc[-1].weight.size(1))).I
        R = torch.tensor(R).float().to(self._device)
        Delta = R @ crs_cor
        model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9 * Delta.float()))
        del R
        return model
        #print(label_list)

    def cali_prototye_model(self,train_loader):
        with torch.no_grad():
            old_vectors, vectors, targets = [], [], []

            pbar = tqdm(enumerate(train_loader), desc='cali_prototye_model', total=len(train_loader), unit='batch')
            for i, batch in pbar:
                (_, data, label) = batch
                images = data.to(self._device)

                old_feature = tensor2numpy(self.old_network_module_ptr(images)["features"])
                feature = tensor2numpy(self._network(images)["features"])

                old_vectors.append(old_feature)
                vectors.append(feature)  
        E_old = np.concatenate(old_vectors)
        E_new = np.concatenate(vectors)
        # 准备训练数据 
        X_tensor = torch.from_numpy(E_old).to(torch.float32)

        y_tensor = torch.from_numpy(E_new).to(torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 划分训练集和测试集
        total_size = len(dataset)
        train_size = int(0.9 * total_size)  # 90% 为训练集
        test_size = total_size - train_size  # 剩余的 10% 为测试集
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # 构造模型参数 && 模型初始化 
        in_features = E_old[0].shape[0]  # 输入维度
        out_dim = E_new[0].shape[0]       # 输出维度 
        calimodel = SimpleNN(in_features,out_dim)
        calimodel = calimodel.to(self._device)
        #calimodel.to(torch.float32) 
        # 设置 学习率 优化器 
        optimizer =optim.SGD(calimodel.parameters(), momentum=0.9, lr=0.01,
            weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,
            eta_min=0)     
        prog_bar = tqdm(range(1000))
        
        # 保存训练过程中的最好模型
        best_loss = float('inf')  # 初始化为无穷大，假设损失越小越好  
        best_model_wts = None  
        logging.info("开始 修正 prototype")
        for _, epoch in enumerate(prog_bar):
            calimodel.train()
            running_loss = 0.0 
            for i, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                #logits = calimodel(inputs)["logits"]
                logits = calimodel(inputs)
                criterion = nn.MSELoss()
                # 计算二次范数损失
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            # 计算每个epoch的平均损失 
            scheduler.step() 
            calimodel.eval()
            test_loss = 0.0
            #correct = 0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = calimodel(inputs)
                    criterion = nn.MSELoss()
                    test_loss += criterion(logits, targets).item() * inputs.size(0)
    
            test_loss /= len(test_dataset)
            if test_loss < best_loss:  
                #print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Best model updated!')  
                best_loss = test_loss  
                best_model_wts = copy.deepcopy(calimodel.state_dict()) 
        logging.info("best_loss: {}".format(best_loss))
        # 选取最好参数 
        calimodel.load_state_dict(best_model_wts)  
        calimodel.eval()
        X_test = torch.from_numpy(np.array(self._means)[:self._known_classes]).to(torch.float64)
        #Y_test = calimodel(X_test.to(self._device))["logits"]
        Y_test = calimodel(X_test.to(self._device))
        Y_test = Y_test.to("cpu")  
        Y_test = Y_test.detach().numpy().tolist()
        self._means[:self._known_classes] = Y_test



class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label





