import numpy as np
from typing import Dict, Optional, Union
from loguru import logger
import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
from ..stereo_matcher.unimatch.UniMatchStereo import UniMatchStereo
from ..stereo_matcher.raft.RAFTStereo import RAFTStereoInference
from ..stereo_matcher.bridgedepth.BridgeDepthStereo import BridgeDepthStereo
from ..stereo_matcher.defom.DefomStereo import DEFOMStereoInference


def pad_len(l: int, pad: int) -> int:
    return int(np.ceil(l / pad)) * pad

def inference_stereo(
        stereo_matcher: Union[UniMatchStereo, RAFTStereoInference, DEFOMStereoInference, BridgeDepthStereo],
        left_img: np.ndarray,
        right_img: np.ndarray,
        pred_mode: str = 'left',
        bidir_verify_th: int = 1,
        max_size: Optional[int] = None,
        padding_factor: int = 32,
        info=None) -> Dict[str, np.ndarray]:

    if info is None:
        info = {}

    pad_fn = lambda x: pad_len(x, padding_factor)

    logger.debug(f"Processing image pair: left shape {left_img.shape}, right shape {right_img.shape}")
    sample = {'left': stereo_matcher.transform_img(left_img), 'right': stereo_matcher.transform_img(right_img)}

    # ensure outputs from transform_img are torch tensors on the matcher device
    if isinstance(sample['left'], np.ndarray):
        left = torch.from_numpy(sample['left']).float().to(stereo_matcher.device).unsqueeze(0)  # [1, 3, H, W]
    else:
        left = sample['left'].to(stereo_matcher.device).unsqueeze(0)  # [1, 3, H, W]

    if isinstance(sample['right'], np.ndarray):
        right = torch.from_numpy(sample['right']).float().to(stereo_matcher.device).unsqueeze(0)  # [1, 3, H, W]
    else:
        right = sample['right'].to(stereo_matcher.device).unsqueeze(0)  # [1, 3, H, W]

    img_max_len = max(left.shape[-2:])
    # resize to nearest size or specified size
    if max_size is None or img_max_len <= max_size:
        # H * W
        inference_size = [pad_fn(left.size(-2)), pad_fn(left.size(-1))]
    else:
        scale_factor = max_size / img_max_len
        inference_size = [pad_fn(left.size(-2) * scale_factor),
                        pad_fn(left.size(-1) * scale_factor)]

    ori_size = left.shape[-2:]
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        left = F.interpolate(left, size=inference_size,
                            mode='bilinear',
                            align_corners=True)
        right = F.interpolate(right, size=inference_size,
                            mode='bilinear',
                            align_corners=True)

    with torch.no_grad():
        if pred_mode == "bidir":
            new_left, new_right = hflip(right), hflip(left)
            left = torch.cat((left, new_left), dim=0)
            right = torch.cat((right, new_right), dim=0)

        if pred_mode == "right":
            left, right = hflip(right), hflip(left)

        pred_disp = stereo_matcher.infer(left, right)
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        # resize back
        pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                mode='bilinear',
                                align_corners=True).squeeze(1)  # [1, H, W]
        pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

    if pred_mode == 'right':
        pred_disp = hflip(pred_disp)
    elif pred_mode == 'bidir':
        pred_disp[1] = hflip(pred_disp[1])
        if bidir_verify_th > 0:
            keep_disp_l = bidir_verify_disp(pred_disp, bidir_verify_th)
    pred_disp = pred_disp.cpu().numpy()
    # 组织结果
    result = {}
    if pred_mode in ['left', 'right']:
        result[f'disparity_{pred_mode}'] = pred_disp[0]
    elif pred_mode == 'bidir':
        disp_l, disp_r = pred_disp
        result[f'disparity_left'] = disp_l
        result[f'disparity_right'] = disp_r

        if bidir_verify_th > 0:
            result[f'disparity_verified'] = keep_disp_l.cpu().numpy()

    return result


def bidir_verify_disp(pred_disp: torch.Tensor,
                    bidir_verify_th: int) -> torch.Tensor:
    """双向验证"""
    disp_l, disp_r = pred_disp
    h, w = disp_r.shape[:2]
    sample_input = rearrange(disp_r, "h w -> 1 1 h w") # sample_input is right disparity
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid_x = grid_x.to(pred_disp.device)
    grid_y = grid_y.to(pred_disp.device)

    sample_grid_x = grid_x - disp_l
    sample_grid_x.clamp_min_(0)
    sample_grid_x = (sample_grid_x - w/2)/(w/2)
    sample_grid_y = (grid_y - h/2)/(h/2)

    sample_grid = torch.stack((sample_grid_x, sample_grid_y), dim=-1)
    sample_grid = rearrange(sample_grid, "h w c -> 1 h w c")
    rec_disp_l = F.grid_sample(sample_input, sample_grid, align_corners=True)
    rec_disp_l = rearrange(rec_disp_l, "1 1 h w -> h w")


    disp_dist = (rec_disp_l - disp_l).abs()

    keep_mask = disp_dist <= bidir_verify_th

    keep_disp_l = disp_l * keep_mask

    return keep_disp_l


