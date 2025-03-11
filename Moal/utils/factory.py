def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        from models.simplecil import Learner
    elif name == "simple_ac":
        from models.simple_ac import Learner
    elif name == "adam_ac":
        from models.adam_ac import Learner
    elif name == "adapt_ac":
        from models.adapt_ac import Learner
    elif name == "adapt_ac_kd":
        from models.adapt_ac_kd import Learner
    elif name == "adapt_ac_kd_com":
        from models.adapt_ac_kd_com import Learner
    elif name == "adapt_ac_kd_com_sdc":
        from models.adapt_ac_kd_com_sdc import Learner
    elif name == "adapt_ac_com_sdc_ema_auto":
        from models.adapt_ac_com_sdc_EMA_auto import Learner
    elif name == "adapt_ac_kd_before_com_sdc":
        from models.adapt_ac_kd_before_com_sdc import Learner
    elif name == "adapt_ac_com_sdc":
        from models.adapt_ac_com_sdc import Learner
    elif name == "adapt_once_ac":
        from models.adapt_once_ac import Learner
    elif name == "pro_ac":
        from models.pro_ac import Learner
    elif name == "pro_ac_fc":
        from models.pro_ac_fc import Learner
    elif name == "pro_ac_kd":
        from models.pro_ac_kd import Learner
    elif name == "pro_ac_single_kd":
        from models.pro_ac_single_kd import Learner
    elif name == "pro_ac_single_kd_com":
        from models.pro_ac_single_kd_com import Learner
    elif name == "pro_ac_single":
        from models.pro_ac_single import Learner
    elif name == "adam_finetune":
        from models.adam_finetune import Learner
    elif name == "adam_ssf":
        from models.adam_ssf import Learner
    elif name == "adam_vpt":
        from models.adam_vpt import Learner 
    elif name == "adam_adapter":
        from models.adam_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "coil":
        from models.coil import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    else:
        assert 0
    
    return Learner(args)
