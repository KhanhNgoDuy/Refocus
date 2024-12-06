import argparse
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import sys
import os 
sys.path.append('Depth-Anything-V2')

from utils.depth_processing import estimate_depth, adaptive_blur

"""
Unified pipeline
Take color image, process depth information, perform blurring & merge
Usage: python run_backend.py --img_path <path-to-color-image>

Notes: uncomment show_plot* to visualize (work for matplotlib)
"""

    
def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    img_path = args.img_path
    depth_config_path = args.depth_config_path
    image = cv2.imread(img_path)
    scene_name = os.path.basename(img_path).split('.')[-2]
    depth = estimate_depth(image, depth_config_path, DEVICE)

    
    #### USER DEFINE ####
    usr_point = (312, 247)
    F = args.f_number
    kernel_t = args.kernel
    output_dir = args.outdir
    #####################

    save_dir = os.path.join(output_dir, scene_name, kernel_t + '_kernel')
    os.makedirs(save_dir, exist_ok=True)

    boked_image = adaptive_blur(image, depth, f_number=F, focal_point=usr_point, scene=scene_name, kernel_type=kernel_t)
    # cv2.circle(boked_image, usr_point, 10, (255, 255, 0), 2)
    cv2.imwrite(f"{save_dir}/output_F-{F}.png", boked_image)
    cv2.imshow("Result", boked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refocus')
    
    parser.add_argument('--img_path', type=str, help='Image to be refocused')
    parser.add_argument('--depth_config_path', type=str, default='./configs/depth_anything_v2.yaml', help='Depth Anything config path')
    parser.add_argument('--f_number', type=float, help='Input your F number (control aperture)') 
    parser.add_argument('--kernel', type=str, choices=['gaussian', 'coc'], help='Kernel type, gaussian or coc (cut-off version)') 
    parser.add_argument('--outdir', type=str, default='./output', help='Kernel type, gaussian or coc (cut-off version)') 
    args = parser.parse_args()

    main(args)