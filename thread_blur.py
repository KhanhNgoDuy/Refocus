import numpy as np
import cv2
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication

from utils import create_gaussian_kernel


def adaptive_blur(color_image, depth_masks, f_number, user_sl_point):
    user_sl_point = (user_sl_point.y(), user_sl_point.x())
    final_img = np.zeros(shape=color_image.shape)
    usr_mask = get_user_select_mask_index(user_sl_point, depth_masks)
    
    for i, mask in enumerate(depth_masks):
        mask_3ch = np.dstack([mask] * 3)
        if usr_mask == i:
            final_img += color_image * mask_3ch.astype(np.uint8)
        else:
            ### BOTTLENECK START
            sigma = 3 * np.abs(usr_mask - i) / f_number
            kernel = create_gaussian_kernel(sigma)
            blur_image = cv2.filter2D(src=color_image, ddepth=-1, kernel=kernel)
            ### BOTTLENECK END
            final_img += blur_image * mask_3ch.astype(np.uint8)
    return final_img


def get_blurred_images(color_image, depth_masks, f_number, user_sl_point):
    assert 1 < f_number <= 22, "Invalid f-number"
    usr_mask = get_user_select_mask_index(user_sl_point, depth_masks)
    blurred_images = []
    
    for i, _ in enumerate(depth_masks):
        sigma = 3 * np.abs(usr_mask - i) / f_number
        kernel = create_gaussian_kernel(sigma)
        blurred_image = cv2.filter2D(src=color_image, ddepth=-1, kernel=kernel)
        blurred_images.append(blurred_image)
    blurred_images = np.array(blurred_images)
    
    return blurred_images

def depth_binning(img_d, num_bins):
    min_intensity, max_intensity = 0, 256
    bin_edges = np.linspace(min_intensity, max_intensity, num_bins + 1)
    masks = [np.logical_and(img_d >= bin_edges[i], img_d < bin_edges[i + 1]) for i in range(num_bins)]
    return masks

def get_user_select_mask_index(coords, masks):
    row, col = coords
    for i, mask in enumerate(masks):
        if mask[row, col]:
            return i
    return -1