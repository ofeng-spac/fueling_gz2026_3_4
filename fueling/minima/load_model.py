import logging
import os
import torch
from copy import deepcopy
from .data_io_roma import DataIOWrapper, lower_config
from .default import get_cfg_defaults
from .romatch import roma_outdoor, tiny_roma_v1_outdoor    
def initialize_matcher(ckpt_path, ckpt2='large'):
    """
    初始化 RoMa 匹配器。
    """
    import sys
    sys.path.append("./third_party/RoMa_minima/")
    

    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if ckpt2 == 'large':

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location=device)
            matcher_model = roma_outdoor(device=device, weights=state_dict)
        else:
            matcher_model = roma_outdoor(device=device)
    else:
        matcher_model = tiny_roma_v1_outdoor(device=device)

    matcher = DataIOWrapper(matcher_model, config=config["test"])
    logging.info(config["test"])

    return matcher.from_cv_imgs