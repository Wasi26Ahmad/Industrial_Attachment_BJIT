import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from prediction import predict_image, load_model

class PredictScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Prediction')
        self.setGeometry(100, 100, 800, 600)

        # Set dark theme
        self.setStyleSheet("background-color: #2D2D2D;")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)

        # Create layout
        layout = QVBoxLayout()

        # File chooser button
        self.file_chooser_button = QPushButton('Choose Image')
        self.file_chooser_button.clicked.connect(self.choose_image)
        layout.addWidget(self.file_chooser_button)

        # Predict button
        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        # Result label
        self.result_label = QLabel('')
        layout.addWidget(self.result_label)

        # Set layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def choose_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if fileName:
            self.image_path = fileName

    def predict(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, 'Error', 'No image selected')
        else:
            result = predict_image(self.image_path, load_model())
            self.result_label.setText(f'Prediction: {result}')

def load_model():
    # Placeholder for loading the model
    pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictScreen()
    window.show()
    sys.exit(app.exec_())
