import os
import cv2
import pickle
from .drawing import vis_disparity
from loguru import logger
from typing import Union
from pathlib import Path

def save_disparity_map(result: dict, output_dir:Union[str, Path]):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for i in range(num):
    if f'disparity_left' in result:
        disp = result[f'disparity_left']  # 获取左视差图
        disp_vis = vis_disparity(disp)
        cv2.imwrite(f"{output_dir}/left_disp.png", disp_vis)
        with open(f"{output_dir}/left_disp.pkl", 'wb') as fp:
            pickle.dump(disp, fp)
        logger.info(f"已保存左视差图: {output_dir}/left_disp.png")

    if f'disparity_right' in result:
        disp = result[f'disparity_right']  # 获取右视差图
        disp_vis = vis_disparity(disp)
        cv2.imwrite(f"{output_dir}/right_disp.png", disp_vis)
        with open(f"{output_dir}/right_disp.pkl", 'wb') as fp:
            pickle.dump(disp, fp)
        logger.info(f"已保存右视差图: {output_dir}/right_disp.png")

    if f'disparity_verified' in result:
        disp = result[f'disparity_verified']  # 获取双向验证后的视差图
        disp_vis = vis_disparity(disp)
        cv2.imwrite(f"{output_dir}/left_disp_verified.png", disp_vis)
        with open(f"{output_dir}/left_disp_verified.pkl", 'wb') as fp:
            pickle.dump(disp, fp)
        logger.info(f"已保存双向验证后的视差图: {output_dir}/left_disp_verified.png")


