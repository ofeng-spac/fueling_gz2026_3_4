import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import hflip
import anyio
from typing import Optional, List, Dict
from .model.unimatch import UniMatch
from loguru import logger

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class UniMatchStereo:
    """异步立体视觉推理类"""

    METHOD = 'UniMatchStereo'

    def __init__(self,
                 weight_path: str,
                 feature_channels: int = 128,
                 num_scales: int = 2,
                 upsample_factor: int = 4,
                 num_head: int = 1,
                 ffn_dim_expansion: int = 4,
                 num_transformer_layers: int = 6,
                 reg_refine: bool = True,
                 task: str = 'stereo',
                 attn_type: str = 'self_swin2d_cross_swin1d',
                 attn_splits_list: List[int] = [2, 8],
                 corr_radius_list: List[int] = [-1, 4],
                 prop_radius_list: List[int] = [-1, 1],
                 num_reg_refine: int = 3,
                 device: str = 'cuda',
                 strict_resume: bool = False):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.attn_type = attn_type
        self.attn_splits_list = attn_splits_list
        self.corr_radius_list = corr_radius_list
        self.prop_radius_list = prop_radius_list
        self.num_reg_refine = num_reg_refine
        self.strict_resume = strict_resume

        # 初始化模型
        self.model = UniMatch(
            feature_channels=feature_channels,
            num_scales=num_scales,
            upsample_factor=upsample_factor,
            num_head=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            num_transformer_layers=num_transformer_layers,
            reg_refine=reg_refine,
            task=task
        ).to(self.device)

        # 加载预训练权重
        self.load_weights(weight_path)

        # 设置为评估模式
        self.model.eval()

        logger.info(f"UniMatchStereo initialized on {self.device}")

        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def load_weights(self, model_path: str):
        """加载预训练模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['model'], strict=self.strict_resume)


    def infer(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model(left, right,
                        attn_type=self.attn_type,
                        attn_splits_list=self.attn_splits_list,
                        corr_radius_list=self.corr_radius_list,
                        prop_radius_list=self.prop_radius_list,
                        num_reg_refine=self.num_reg_refine,
                        task='stereo',
                        )['flow_preds'][-1].detach()
