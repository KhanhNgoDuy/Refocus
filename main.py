import cv2
import numpy as np
import argparse
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QPushButton
from PyQt5.uic import loadUi
from time import perf_counter as pc, sleep

from thread_image import ImageThread
# from thread_blur import ImageProcessingThread
from thread_depthanything import get_depth
from click_label import ClickLabel
from utils import create_gaussian_kernel, create_soft_coc_kernel


class MainWindow(QMainWindow):
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self, path, f_num, kernel_t):
        super().__init__()

        # Set up UI
        self.ui = loadUi('new.ui', self)
        self.label = self.findChild(ClickLabel, "label")

        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # H, W, C = self.image.shape
        H, W = 512, 1024
        self.image = cv2.resize(self.image, (W, H))
        
        # self.image_depth = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE)
        self.image_depth = get_depth(path)
        self.image_depth = cv2.resize(self.image_depth, (W, H))

        self.setFixedWidth(W)
        self.setFixedHeight(H)

        # Create attributes
        self.f_num = f_num
        self.offset_point = QPoint(0, 0)
        self.kernel_type= kernel_t

        # Connect the label's clicked signal
        self.label.clicked.connect(self.handle_label_click)

    @pyqtSlot(QPoint)
    def handle_label_click(self, pos):
        """Handle the click on the label and trigger adaptive blur."""
        # start_sum = pc()
        self.offset_point = pos + QPoint(0, 28)  # bug
        # start_depth_binning = pc()
        depth_masks = self.depth_binning(self.image_depth, num_bins=8)
        # end_depth_binning = pc()
        # start_adaptive_blur = pc()
        blurred_image = self.adaptive_blur(depth_masks).astype(np.uint8)
        # end_adaptive_blur = pc()
        # start_display_image = pc()
        self.display_image(blurred_image)
        # end_display_image = pc()
        # end_sum = pc()
        # print("Sum: ", end_sum - start_sum)
        # print("depth_binning: ", end_depth_binning - start_depth_binning)
        # print("adaptive_blur: ", end_adaptive_blur - start_adaptive_blur)
        # print("display_image: ", end_display_image - start_display_image)

    def display_image(self, rgb_image):
        """Convert and display an RGB image on the label."""
        self.image_ready.emit(rgb_image)
        # print(f"RGB: {rgb_image.shape}")
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
        
        for i, mask in enumerate(depth_masks):
            mask_3ch = np.dstack([mask] * 3)
            if usr_mask == i:
                final_img += self.image * mask_3ch.astype(np.uint8)
            else:
                ###3 These number are assumptions
                if self.kernel_type == 'gaussian':
                    sigma = 3 * np.abs(usr_mask - i) / self.f_num
                    kernel = create_gaussian_kernel(sigma)
                elif self.kernel_type == 'coc':
                    coc_radius = 3 + 2 * int(np.abs(usr_mask -i) / self.f_num)
                    c = 4.0
                    kernel = create_soft_coc_kernel(coc_radius, falloff=c)
                else:
                    raise NotImplementedError(f"Unknown kernel type {self.kernel_type}")
                blur_image = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
                final_img += blur_image * mask_3ch.astype(np.uint8)
        return final_img

    @staticmethod
    def depth_binning(img_d, num_bins):
        min_intensity, max_intensity = 0, 256
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
    parser.add_argument('--kernel', type=str,choices=['gaussian', 'coc'], default='coc', help='Kernel type')
    args = parser.parse_args()
    app = QApplication([])
    window = MainWindow(args.img_path, args.F, args.kernel)
    window.show()
    app.exec_()
