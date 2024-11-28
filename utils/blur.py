import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils.kernel import create_soft_coc_kernel


def adaptive_blur(color_image, depth_masks, f_number, focal_point):
    assert 1 < f_number <= 22, "Don't use such a weird lense"
    final_img = np.zeros(shape=color_image.shape)    
    usr_mask = get_user_select_mask_index(focal_point, depth_masks)
    print(f"User selected mask index: {usr_mask}")
    for i, mask, in enumerate(depth_masks):
        mask_3ch = np.dstack([mask] * 3)
        if usr_mask == i:
            final_img += color_image * mask_3ch.astype(np.uint8)
        else:
            # delta depth can be considered as the difference in depth bins index

            # TODO: coc radius depend on image size
            # TODO: gamma correction to magnify high-intensity px
            coc_radius = 10 * np.abs(usr_mask -i) / f_number
            c = 4.0
            coc_kernel = create_soft_coc_kernel(coc_radius, falloff=c)
            plt.imshow(coc_kernel, cmap='gray')
            plt.savefig(f'soft_coc_kernel_D0-{coc_radius}_c-{c}.png')
            blur_image = cv2.filter2D(src=color_image, ddepth=-1, kernel=coc_kernel)
            final_img += blur_image * mask_3ch.astype(np.uint8)
    return final_img
    

def get_user_select_mask_index(coords, masks):
    row, col = coords
    for i, mask in enumerate(masks):
        if mask[row, col]:
            return i
    return -1
