from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("GUI.ui", self)  # Path disesuaikan jika GUI.ui ada di folder yang sama

        # Ambil komponen dari GUI
        self.label1: QLabel = self.findChild(QLabel, "label")
        self.label2: QLabel = self.findChild(QLabel, "label_2")
        self.button1: QPushButton = self.findChild(QPushButton, "pushButton")
        self.button2: QPushButton = self.findChild(QPushButton, "pushButton_2")

        # Hubungkan tombol ke fungsi
        self.button1.clicked.connect(self.load_gambar1)
        self.button2.clicked.connect(self.load_gambar2)

    def load_gambar1(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar 1", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            pixmap = QPixmap(path).scaled(self.label1.width(), self.label1.height())
            self.label1.setPixmap(pixmap)

    def load_gambar2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar 2", "", "Image Files (*.png *.jpg *.bmp)")
        if path:
            pixmap = QPixmap(path).scaled(self.label2.width(), self.label2.height())
            self.label2.setPixmap(pixmap)
