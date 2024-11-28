import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.others import load_config_file, show_plot
from utils.kernel import create_soft_coc_kernel
from depth_anything_v2.dpt import DepthAnythingV2


def estimate_depth(color_image, depth_config, device='cuda'):
    depth_configs = load_config_file(depth_config)
    encoder = depth_configs['encoder']
    depth_anything = DepthAnythingV2(**depth_configs['model_configs'][encoder])
    depth_anything.load_state_dict(torch.load(depth_configs['checkpoint'], map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()

    estimated_depth = depth_anything.infer_image(color_image)
    # Visualize and check for depth values
    show_plot(estimated_depth, title='Depth', cmap='gray')
    # Current ISSUE: depth map is inversed depth
    return estimated_depth


def depth_binning(img_d, num_bins):
    bin_edges = np.linspace(img_d.min(), img_d.max(), num_bins + 1)
    # Create a list to store the binary masks
    masks = []

    # Iterate over the bin edges and create masks
    for i in range(num_bins):
        mask = np.logical_and(img_d >= bin_edges[i], img_d < bin_edges[i+1])
        masks.append(mask)
        # show_plot(mask, title=f"Mask {i}", cmap='gray')

    return masks


def adaptive_blur(color_image, depth_image, f_number, focal_point):
    assert 1 < f_number <= 22, "Don't use such a weird lense"
    # Get depth masks first
    depth_masks = depth_binning(depth_image, num_bins=16)

    final_img = np.zeros(shape=color_image.shape, dtype=np.uint8)    
    usr_mask = get_user_select_mask_index(focal_point, depth_masks)
    print(f"User selected mask index: {usr_mask}")
    for i, mask, in enumerate(depth_masks):
        mask_3ch = np.dstack([mask] * 3)
        if usr_mask == i:
            # User select <=> focused plane, no blur
            patch = color_image * mask_3ch.astype(np.uint8)
            # show_plot(patch, title='Focused patch')
            final_img += patch
        else:
            # delta depth can be considered as the difference in depth bins index

            # TODO: coc radius depend on image size
            # TODO: gamma correction to magnify high-intensity px
            coc_radius = 3 + np.abs(usr_mask -i) / f_number
            c = 4.0
            coc_kernel = create_soft_coc_kernel(coc_radius, falloff=c)

            # show_plot(coc_kernel, title=f"Kernel radius: {coc_radius}, falloff: {c}", cmap='gray')

            blur_image = cv2.filter2D(src=color_image, ddepth=-1, kernel=coc_kernel)
            patch =  blur_image * mask_3ch.astype(np.uint8)
            # show_plot(blur_image, title=f"Blur patch {i}")
            final_img += patch
            
    return final_img


def get_user_select_mask_index(coords, masks):
    row, col = coords
    for i, mask in enumerate(masks):
        if mask[row, col]:
            return i
    return -1
