'''
# gmstereo-scale2-regrefine3 model
CUDA_VISIBLE_DEVICES=0 python main_stereo.py \
--inference_dir demo/stereo-middlebury \
--inference_size 1024 1536 \
--output_path output/gmstereo-scale2-regrefine3-middlebury \
--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3

# optionally predict both left and right disparities
#--pred_bidir_disp

'''
import os
import os.path as osp
import argparse
from glob import glob
import numpy as np
import re

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import hflip
from einops import rearrange
import pickle
import time
from PIL import Image
import cv2

import pylab as plt

from utils.visualization import vis_disparity
from unimatch.unimatch import UniMatch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def pad_len(l, pad):
    return int(np.ceil(l / pad)) * pad

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--padding_factor', default=16, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')

    parser.add_argument('--task', default='stereo', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='self_swin2d_cross_1d', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # evaluation
    parser.add_argument('--max_size', default=None, type=int)
    parser.add_argument('--save_vis_disp', action='store_true')

    parser.add_argument('--output_path', default='output', type=str)

    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # inference
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_dir_left', default=None, type=str)
    parser.add_argument('--inference_dir_right', default=None, type=str)
    parser.add_argument('--pred_mode', choices=("left", "right", "bidir"), default="left")
    parser.add_argument('--bidir_verify_th', type=int, default=0)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    inference_dir = args.inference_dir
    inference_dir_left = args.inference_dir_left
    inference_dir_right = args.inference_dir_right
    output_path = args.output_path
    padding_factor = args.padding_factor
    max_size = args.max_size
    attn_type = args.attn_type
    attn_splits_list = args.attn_splits_list
    corr_radius_list = args.corr_radius_list
    prop_radius_list = args.prop_radius_list
    num_reg_refine = args.num_reg_refine
    pred_mode = args.pred_mode
    bidir_verify_th = args.bidir_verify_th if pred_mode == "bidir" else 0
    debug = args.debug

    if debug:
        print(args)

    check_path(output_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task).to(device)

    if debug:
        print(model)

    num_params = sum(p.numel() for p in model.parameters())
    if debug:
        print('=> Number of trainable parameters: %d' % num_params)

    if args.resume:
        print("=> Load checkpoint: %s" % args.resume)

        loc = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)

        model.load_state_dict(checkpoint['model'], strict=args.strict_resume)

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    assert inference_dir or (inference_dir_left and inference_dir_right)

    if inference_dir is not None:
        filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))

        left_filenames = filenames[::2]
        right_filenames = filenames[1::2]

    else:
        left_filenames = sorted(glob(inference_dir_left + '/*.png') + glob(inference_dir_left + '/*.jpg'))
        right_filenames = sorted(glob(inference_dir_right + '/*.png') + glob(inference_dir_right + '/*.jpg'))

    assert len(left_filenames) == len(right_filenames)

    # 从inference_dir提取光源类型和设备编号
    light_type = None
    device_id = None
    if inference_dir is not None:
        # 匹配格式如 orbbec_output/Flood_light/device0
        match = re.search(r'(Flood_light|Laser_light)/device(\d+)', inference_dir)
        if match:
            light_type = match.group(1)
            device_id = match.group(2)
            print(f"检测到光源类型: {light_type}, 设备编号: {device_id}")
        else:
            print("警告: 无法从inference_dir中提取光源类型和设备编号")

    # 创建对应的输出子目录
    output_subdir = output_path
    if light_type and device_id:
        output_subdir = os.path.join(output_path, light_type, f"device{device_id}")
        check_path(output_subdir)
    else:
        output_subdir = output_path

    num_samples = len(left_filenames)
    print('%d test samples found' % num_samples)

    for i in range(num_samples):

        if (i + 1) % 50 == 0:
            print('predicting %d/%d' % (i + 1, num_samples))

        left_name = left_filenames[i]
        right_name = right_filenames[i]

        left_img = cv2.imread(left_name, cv2.IMREAD_COLOR)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.imread(right_name, cv2.IMREAD_COLOR)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        sample = {'left': transform(left_img), 'right': transform(right_img)}

        left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
        right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]

        img_max_len = max(left.shape[-2:])
        # resize to nearest size or specified size
        if max_size is None or img_max_len <= max_size:
            # H * W
            inference_size = [pad_len(left.size(-2), padding_factor), pad_len(left.size(-1), padding_factor)]
        else:
            scale_factor = max_size / img_max_len
            inference_size = [pad_len(left.size(-2) * scale_factor, padding_factor),
                              pad_len(left.size(-1) * scale_factor, padding_factor)]

        ori_size = left.shape[-2:]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            left = F.interpolate(left, size=inference_size,
                                 mode='bilinear',
                                 align_corners=True)
            right = F.interpolate(right, size=inference_size,
                                  mode='bilinear',
                                  align_corners=True)

        with torch.no_grad():
            if pred_mode == "right":
                left, right = hflip(right), hflip(left)
            elif pred_mode == 'bidir':
                new_left, new_right = hflip(right), hflip(left)
                left = torch.cat((left, new_left), dim=0)
                right = torch.cat((right, new_right), dim=0)

            pred_disp = model(left, right,
                              attn_type=attn_type,
                              attn_splits_list=attn_splits_list,
                              corr_radius_list=corr_radius_list,
                              prop_radius_list=prop_radius_list,
                              num_reg_refine=num_reg_refine,
                              task='stereo',
                              )['flow_preds'][-1]  # [1, H, W]

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
            disp_l, disp_r = pred_disp
            h, w = disp_r.shape[:2]
            sample_input = rearrange(disp_r, "h w -> 1 1 h w") # sample_input is right disparity
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
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

            if debug:
                plt.figure("disp_l")
                plt.imshow(disp_l.cpu().numpy())
                plt.figure("disp_r")
                plt.imshow(disp_r.cpu().numpy())
                plt.figure("rec_disp_l")
                plt.imshow(rec_disp_l.cpu().numpy())
                plt.figure("disp_dist")
                plt.imshow(disp_dist.cpu().numpy())
                plt.figure("keep_disp_l")
                plt.imshow(keep_disp_l.cpu().numpy())
                plt.show()

        pred_disp = pred_disp.cpu().numpy()
        # save_prefix =  '-'.join(os.path.join(output_path, osp.splitext(osp.basename(left_name))[0]).rsplit('-')[:-1]) # remove '_l', '_r' posfix
        save_prefix_left = os.path.splitext(os.path.join(output_subdir, os.path.basename(left_name)))[0]
        save_prefix_right = os.path.splitext(os.path.join(output_subdir, os.path.basename(right_name)))[0]
        print(f"保存路径: {save_prefix_left}")
        pkl_file_path_left = f"{save_prefix_left}-disp_l.pkl"
        pkl_file_path_right = f"{save_prefix_right}-disp_r.pkl"
        pkl_file_path_keep_disp_l = f"{save_prefix_left}-disp_keep_disp_l.pkl"

        if pred_mode == 'left' or pred_mode == "right":
            disp = pred_disp[0]
            posfix = pred_mode[0]
            if pred_mode == 'left':
                with open(pkl_file_path_left, 'wb') as fp:
                    pickle.dump(disp, fp)
                cv2.imwrite(save_prefix_left + f"-disp.png", disp.astype('u2'))
                cv2.imwrite(save_prefix_left + f"-disp_vis_{posfix}.png", vis_disparity(disp))

            if pred_mode == 'right':
                with open(pkl_file_path_right, 'wb') as fp:
                    pickle.dump(disp, fp)
                cv2.imwrite(save_prefix_right + f"-disp.png", disp.astype('u2'))
                cv2.imwrite(save_prefix_right + f"-disp_vis_{posfix}.png", vis_disparity(disp))

            if debug:
                plt.figure(f"L {left_name}")
                plt.imshow(left_img)

                plt.figure(f"R {right_name}")
                plt.imshow(right_img)

                plt.figure(f"disp")
                plt.imshow(disp)
                plt.show()

        elif pred_mode == "bidir":
            disp_l, disp_r = pred_disp
            keep_disp_l = keep_disp_l.cpu().numpy()
            print(type(disp_l))  # 确保是 numpy 数组
            print(disp_l.shape)  # 确保形状正确

            with open(pkl_file_path_left, 'wb') as fp:
                pickle.dump(disp_l, fp)
            # with open(pkl_file_path_right, 'wb') as fp:
            #     pickle.dump(disp_r, fp)
            with open(pkl_file_path_keep_disp_l, 'wb') as fp:
                pickle.dump(keep_disp_l, fp)

            cv2.imwrite(save_prefix_left + f"-disp_l.png", disp_l.astype('u2'))
            # cv2.imwrite(save_prefix_right + f"-disp_r.png", disp_r.astype('u2'))
            cv2.imwrite(save_prefix_left + f"-keep_disp_l.png", keep_disp_l.astype('u2'))
            cv2.imwrite(save_prefix_left + f"-disp_vis_l.png", vis_disparity(disp_l))
            # cv2.imwrite(save_prefix_right + f"-disp_vis_r.png", vis_disparity(disp_r))
            cv2.imwrite(save_prefix_left + f"-keep_disp_vis_l.png", vis_disparity(keep_disp_l))

            if debug:
                plt.figure(f"L {left_name}")
                plt.imshow(left_img)

                plt.figure(f"R {right_name}")
                plt.imshow(right_img)

                plt.figure(f"disp_l")
                plt.imshow(disp_l)
                plt.figure(f"disp_r")
                plt.imshow(disp_r)
                plt.show()

    print('Done!')

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")
