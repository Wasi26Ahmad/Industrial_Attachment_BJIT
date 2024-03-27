import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, \
    QWidget, QMessageBox, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor, QPixmap
import torch
from torchvision import models

from prediction import predict_image, load_model


class PredictScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Prediction')
        self.setGeometry(100, 100, 800, 600)

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

        main_layout = QVBoxLayout()

        self.image_label = QLabel()
        main_layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)

        self.file_chooser_button = QPushButton('Choose Image')
        self.file_chooser_button.clicked.connect(self.choose_image)
        self.file_chooser_button.setFixedSize(300, 50)
        self.file_chooser_button.setStyleSheet("background-color: #000208; color: white;")
        button_layout.addWidget(self.file_chooser_button)

        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.predict)
        self.predict_button.setFixedSize(300, 50)
        self.predict_button.setStyleSheet("background-color: #000208; color: white;")
        button_layout.addWidget(self.predict_button)

        button_layout.addStretch(1)

        bottom_layout = QVBoxLayout()
        bottom_layout.addStretch(1)
        bottom_layout.addLayout(button_layout)

        main_layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def choose_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.xpm *.jpg);;All Files (*)",
                                                  options=options)
        if fileName:
            self.image_path = fileName
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))

    def predict(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, 'Error', 'No image selected')
        else:
            result = predict_image(self.image_path, load_model())
            dialog = QDialog(self)
            dialog.setWindowTitle('Prediction Result')
            dialog.setGeometry(100, 100, 300, 200)
            layout = QVBoxLayout()

            layout.addStretch(1)
            result_label = QLabel(f'Prediction: {result}')
            result_label.setStyleSheet("color: white; font-size: 36px;")
            layout.addWidget(result_label)
            layout.addStretch(1)

            dialog.setLayout(layout)
            dialog.exec_()


def load_model():
    try:
        model = models.resnet18()
        model.load_state_dict(torch.load('model_attachment_V1.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictScreen()
    window.show()
    sys.exit(app.exec_())
