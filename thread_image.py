import numpy as np
import cv2
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication


class ImageThread(QObject):
    image_signal = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, path='images/room.png'):
        super().__init__()
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        H, W, C = self.image.shape
        self.image = cv2.resize(self.image, (W // 2, H // 2))

    def run(self):
        while True:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            self.image_signal.emit(self.image)
            
        print("stopped")
        self.finished.emit()