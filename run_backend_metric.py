import argparse
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import sys
sys.path.append('Depth-Anything-V2')

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from utils.utils import load_config_file

"""
Model to give depth map in meter, true to scale
Originally was thought to be the solver but results are just pain in the ass
Not used
"""

    
def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    img_path = args.img_path
    depth_config_path = args.depth_config_path

    depth_configs = load_config_file(depth_config_path)
    encoder = depth_configs['encoder']
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20
    
    depth_anything = DepthAnythingV2(**{**depth_configs['model_configs'][encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/ckpts_metric/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    image = cv2.imread(img_path)
    depth = depth_anything.infer_image(image, depth_configs['input_size'])

    # Visualize and check for depth values
    plt.subplot(121)
    plt.imshow(image)
    plt.title("RGB")
    plt.subplot(122)
    plt.imshow(depth)
    plt.title("Depth")
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refocus')
    
    parser.add_argument('--img_path', type=str, help='Image to be refocused')
    parser.add_argument('--depth_config_path', type=str, default='./configs/depth_anything_v2.yaml', help='Depth Anything config path')
    args = parser.parse_args()

    main(args)