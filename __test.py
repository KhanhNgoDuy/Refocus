from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Discrete Non-Linear QSlider Example")
        
        # Define discrete non-linear values
        self.non_linear_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]

        # Create the slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.non_linear_values) - 1)  # Map to indices
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        
        # Create a label to display the current value
        self.label = QLabel("Value: 1")

        # Connect slider value change to the update function
        self.slider.valueChanged.connect(self.update_label)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_label(self, value):
        # Map slider value to non-linear value
        mapped_value = self.non_linear_values[value]
        self.label.setText(f"Value: {mapped_value}")

# Run the application
app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
