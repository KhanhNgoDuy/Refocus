import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread, pyqtSlot, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QPushButton
from PyQt5.uic import loadUi
# from time import perf_counter as pc, sleep

from thread_image import ImageThread
# from thread_blur import ImageProcessingThread
from click_label import ClickLabel
from utils import create_gaussian_kernel


class MainWindow(QMainWindow):
    image_ready = pyqtSignal(np.ndarray)

    def __init__(self, path='images/whale.jpg'):
        super().__init__()

        # Set up UI
        self.ui = loadUi('new.ui', self)
        self.label = self.findChild(ClickLabel, "label")

        path = 'images/nguyen.png'
        path_depth = 'images/nguyen_depth.png'

        self.image = cv2.imread(path, cv2.COLOR_RGB2BGR)
        H, W, C = self.image.shape
        self.image = cv2.resize(self.image, (W // 2, H // 2))
        
        self.image_depth = cv2.imread(path_depth, cv2.IMREAD_GRAYSCALE)
        self.image_depth = cv2.resize(self.image_depth, (W // 2, H // 2))
        
        print(f"Depth: {self.image_depth.shape}")

        self.setFixedWidth(W // 2)
        self.setFixedHeight(H // 2)

        # Create attributes
        self.f_num = 2
        self.offset_point = QPoint(0, 0)

        # Connect the label's clicked signal
        self.label.clicked.connect(self.handle_label_click)

    @pyqtSlot(QPoint)
    def handle_label_click(self, pos):
        """Handle the click on the label and trigger adaptive blur."""
        self.offset_point = pos + QPoint(0, 28)  # bug
        depth_masks = self.depth_binning(self.image_depth, num_bins=8)
        blurred_image = self.adaptive_blur(self.image, depth_masks, self.f_num, self.offset_point).astype(np.uint8)
        self.display_image(blurred_image)

    def display_image(self, rgb_image):
        """Convert and display an RGB image on the label."""
        self.image_ready.emit(rgb_image)
        print(f"RGB: {rgb_image.shape}")
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
    
    def adaptive_blur(self, color_image, depth_masks, f_number, user_sl_point):
        assert 1 < f_number <= 22, "Invalid f-number"
        user_sl_point = (user_sl_point.y(), user_sl_point.x())
        final_img = np.zeros(shape=color_image.shape)
        usr_mask = self.get_user_select_mask_index(user_sl_point, depth_masks)
        for i, mask in enumerate(depth_masks):
            mask_3ch = np.dstack([mask] * 3)
            if usr_mask == i:
                final_img += color_image * mask_3ch.astype(np.uint8)
            else:
                sigma = 3 * np.abs(usr_mask - i) / f_number
                kernel = create_gaussian_kernel(sigma)
                blur_image = cv2.filter2D(src=color_image, ddepth=-1, kernel=kernel)
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
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
