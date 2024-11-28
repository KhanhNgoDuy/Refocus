import argparse
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import sys
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
    depth = estimate_depth(image, depth_config_path, DEVICE)
    
    #### USER DEFINE ####
    usr_point = (165, 100)
    F = 2.0
    #####################

    boked_image = adaptive_blur(image, depth, f_number=F, focal_point=usr_point)
    cv2.circle(boked_image, usr_point, 10, (255, 255, 0), 2)
    cv2.imshow("Result", boked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refocus')
    
    parser.add_argument('--img_path', type=str, help='Image to be refocused')
    parser.add_argument('--depth_config_path', type=str, default='./configs/depth_anything_v2.yaml', help='Depth Anything config path')
    args = parser.parse_args()

    main(args)