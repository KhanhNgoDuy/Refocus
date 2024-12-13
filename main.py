import cv2
import numpy as np
import argparse
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QPushButton, QSlider
from PyQt5.uic import loadUi
from time import perf_counter as pc, sleep
import math

from thread_image import ImageThread
# from thread_blur import ImageProcessingThread
from thread_depthanything import get_depth
from click_label import ClickLabel


class MainWindow(QMainWindow):
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self, path, f_num, kernel_t):
        super().__init__()

        # Set up UI
        self.ui = loadUi('new.ui', self)
        self.label = self.findChild(ClickLabel, "label")
        # self.slider_src = self.findChild(QSlider, "slider_src")
        self.slider_tgt = self.findChild(QSlider, "slider_tgt")
        self.button_save = self.findChild(QPushButton, "button_save")

        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        H, W, C = self.image.shape
        self.orig_H = H
        self.orig_W = W
        fixed_res = 756
        res = max(H, W)
        scale = fixed_res / res
        H, W = int(H*scale), int(W*scale)
        self.image = cv2.resize(self.image, (W, H))
        
        # self.image_depth = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE)
        self.image_depth = get_depth(path)
        self.image_depth = cv2.resize(self.image_depth, (W, H))

        self.setFixedWidth(W)
        self.setFixedHeight(H)

        # Create attributes
        self.f_num = f_num
        self.offset_point = QPoint(0, 0)
        self.kernel_type = kernel_t
        self.display_image(self.image)
        self.current_image = self.image

        # Define non-linear value mappings for the sliders
        self.slider_values = [1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 9.0, 10.0, 11.0, 13.0, 16.0]

        # Configure sliders
        # self.configure_slider(self.slider_src, self.slider_values, self.update_src_label)
        self.configure_slider(self.slider_tgt, self.slider_values, self.update_tgt_label)

        # Connect the label's clicked signal
        self.label.clicked.connect(self.handle_label_click)
        self.button_save.clicked.connect(self.save_image)

    def configure_slider(self, slider, value_list, update_function):
        slider.setMinimum(0)
        slider.setMaximum(len(value_list) - 1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(lambda val: update_function(val, value_list))

    def update_tgt_label(self, value, value_list):
        mapped_value = value_list[value]
        print(f"Changed target f-number to: {mapped_value}")
        self.f_num = mapped_value
        self.update_image()

    @pyqtSlot(QPoint)
    def handle_label_click(self, pos):
        """Handle the click on the label and trigger adaptive blur."""
        self.offset_point = pos + QPoint(0, 28)  # bug
        self.update_image()

    def save_image(self):
        """Save the currently displayed image."""
        save_path = f"out_{self.f_num}_{self.offset_point.x(), self.offset_point.y()}.png"
        rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        rgb_orig_size = cv2.resize(rgb_image, (self.orig_W, self.orig_H))

        cv2.imwrite(save_path, rgb_orig_size)
        print(f"Image saved at {save_path}")

    def update_image(self):
        depth_masks = self.depth_binning(self.image_depth, num_bins=16)
        blurred_image = self.adaptive_blur(depth_masks).astype(np.uint8)
        self.display_image(blurred_image)

    def display_image(self, rgb_image):
        """Convert and display an RGB image on the label."""
        self.image_ready.emit(rgb_image)
        # print(f"RGB: {rgb_image.shape}")
        self.current_image = rgb_image
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        pen = QPen(Qt.green)
        pen.setWidth(5)

        painter = QPainter(pixmap)
        painter.setPen(pen)
        painter.drawPoint(self.offset_point)
        painter.end()

        self.label.setPixmap(pixmap)
    
    def adaptive_blur(self, depth_masks):
        assert 1 < self.f_num <= 22, "Invalid f-number"
        user_sl_point = (self.offset_point.y(), self.offset_point.x())
        final_img = np.zeros(shape=self.image.shape)
        usr_mask = self.get_user_select_mask_index(user_sl_point, depth_masks)
        min_d, max_d = self.image_depth.min(), self.image_depth.max()
        
        def get_real_depth(d):
            return 1 + (d - min_d) * (50) / (max_d - min_d)
        
        f = 0.035  # focal length in meters (35 mm)
        N = self.f_num
        
        # Get focus depth once
        focus_depth = get_real_depth(self.image_depth[depth_masks[usr_mask]].mean())
        
        for i, mask in enumerate(depth_masks):
            mask_3ch = np.dstack([mask] * 3)
            if usr_mask == i:
                final_img += self.image * mask_3ch.astype(np.uint8)
            else:
                current_depth = get_real_depth(self.image_depth[mask].mean())
                
                # Calculate CoC
                numerator = (f * f * abs(focus_depth - current_depth))
                denominator = (current_depth * (focus_depth - f))
                CoC = abs((f/N) * (numerator / denominator))
                CoC = CoC * 2.5
                
                # Convert CoC to pixels
                sensor_width = 0.036
                image_width = self.image.shape[1]
                pixels_per_meter = image_width / sensor_width
                k = 60  # scaling constant

                CoC_pixels = CoC * pixels_per_meter * k
                CoC_pixels = np.clip(CoC_pixels, 0, 60)
                radius = CoC_pixels / 2

                # Only create and apply blur if the radius is significant
                # And the threshold increase with F-stop -> small aparture don't blur
                threshold = 0.5 * (N/8.0)
                if radius > threshold:
                    # Create bilateral depth kernel
                    kernel_size = int(2 * math.ceil(radius) + 1)
                    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
                    spatial_kernel = np.exp(-(x*x + y*y) / (2 * radius * radius))
                    depth_kernel = np.exp(-abs(current_depth - focus_depth) / (2 * radius))
                    kernel = spatial_kernel * depth_kernel
                    kernel = kernel / kernel.sum()
                    
                    blur_image = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
                    blur_image = cv2.bilateralFilter(blur_image.astype(np.uint8), 9, 75, 75)
                else:
                    blur_image = self.image

                final_img += blur_image * mask_3ch
        
        return final_img.astype(np.uint8)

    @staticmethod
    def depth_binning(img_d, num_bins):
        min_intensity, max_intensity = img_d.min(), img_d.max()
        bin_edges = np.linspace(min_intensity, max_intensity, num_bins + 1)
        masks = [np.logical_and(img_d >= bin_edges[i], img_d < bin_edges[i + 1]) for i in range(num_bins)]
        return masks

    @staticmethod
    def get_user_select_mask_index(coords, masks):
        row, col = coords
        for i, mask in enumerate(masks):
            if mask[row, col]:
                return i
        return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='Path to image to be refocused')
    parser.add_argument('--F', type=float, required=True, help='Desired F-number')
    args = parser.parse_args()
    app = QApplication([])
    window = MainWindow(args.img_path, args.F, 'coc')  # Removed kernel choice
    window.show()
    app.exec_()
