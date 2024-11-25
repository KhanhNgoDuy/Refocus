import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QPoint, pyqtSignal


class ClickLabel(QLabel):
    clicked = pyqtSignal(QPoint)  # Signal to emit clicked position
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point = QPoint(0, 0)

    def mousePressEvent(self, event):
        pos = event.pos()
        # self.points.append(pos)
        self.point = pos
        self.clicked.emit(pos)

    # def mouseMoveEvent(self, event):
    #     pos = event.pos()
    #     self.points.append(pos)

    # def resizeEvent(self, a0) -> None:
    #     print(self.width())