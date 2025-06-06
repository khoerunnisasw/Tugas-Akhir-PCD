import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QTabWidget, QTextEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from skimage.feature import graycomatrix, graycoprops

# ---------------------- FUNGSI EKSTRAKSI ----------------------

def ekstrak_warna(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = np.mean(img_rgb, axis=(0, 1))
    std = np.std(img_rgb, axis=(0, 1))
    return {'mean': mean.tolist(), 'std': std.tolist()}

def ekstrak_bentuk(image_path):
    img = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        return {'area': area, 'perimeter': perimeter}
    return {'area': 0, 'perimeter': 0}

def ekstrak_tekstur(image_path):
    img = cv2.imread(image_path, 0)
    glcm = graycomatrix(img, [1], [0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -1 * (glcm * np.log2(glcm + 1e-10)).sum()
    return {'contrast': contrast, 'entropy': entropy}

# ---------------------- GUI ----------------------

class SampahClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Sampah Berbasis Warna, Bentuk, dan Tekstur")
        self.setGeometry(100, 100, 1000, 700)
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)

        title_label = QLabel("🗑️ Klasifikasi Sampah Digital")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Load Image
        load_layout = QHBoxLayout()
        self.load_btn = QPushButton("📂 Muat Gambar")
        self.load_btn.clicked.connect(self.load_image)

        self.image_label = QLabel("Preview Gambar")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("""
            border: 2px dashed #95a5a6;
            background-color: #ecf0f1;
            border-radius: 10px;
        """)
        self.image_label.setAlignment(Qt.AlignCenter)

        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.image_label)
        main_layout.addLayout(load_layout)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_warna_tab(), "🌈 Warna")
        self.tabs.addTab(self.create_bentuk_tab(), "🔵 Bentuk")
        self.tabs.addTab(self.create_tekstur_tab(), "🧵 Tekstur")
        main_layout.addWidget(self.tabs)

        # Klasifikasi
        classify_layout = QHBoxLayout()
        self.classify_btn = QPushButton("🚀 Klasifikasi Sampah")
        self.classify_btn.clicked.connect(self.handle_klasifikasi)

        self.result_label = QLabel("Kategori: -")
        self.result_label.setStyleSheet("font-size: 20px; font-weight: bold; color: blue;")

        classify_layout.addWidget(self.classify_btn)
        classify_layout.addWidget(self.result_label)
        main_layout.addLayout(classify_layout)

        # Footer
        footer_layout = QHBoxLayout()
        self.reset_btn = QPushButton("🔄 Reset")
        self.exit_btn = QPushButton("❌ Keluar")
        self.reset_btn.clicked.connect(self.reset_all)
        self.exit_btn.clicked.connect(self.close)
        footer_layout.addWidget(self.reset_btn)
        footer_layout.addWidget(self.exit_btn)
        main_layout.addLayout(footer_layout)

        self.setLayout(main_layout)

        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                background-color: #f5f6fa;
            }
            QLabel#TitleLabel {
                font-size: 26px;
                font-weight: bold;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #ccc;
                padding: 6px;
                border-radius: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #aaa;
                padding: 6px;
            }
        """)

    def create_warna_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.warna_btn = QPushButton("Ekstrak Warna")
        self.warna_btn.clicked.connect(self.handle_ekstrak_warna)
        self.warna_features = QTextEdit()
        self.warna_features.setReadOnly(True)
        layout.addWidget(self.warna_btn)
        layout.addWidget(QLabel("Fitur Warna (mean, std RGB):"))
        layout.addWidget(self.warna_features)
        tab.setLayout(layout)
        return tab

    def create_bentuk_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.bentuk_btn = QPushButton("Ekstrak Bentuk")
        self.bentuk_btn.clicked.connect(self.handle_ekstrak_bentuk)
        self.bentuk_features = QTextEdit()
        self.bentuk_features.setReadOnly(True)
        layout.addWidget(self.bentuk_btn)
        layout.addWidget(QLabel("Fitur Bentuk (luas, keliling):"))
        layout.addWidget(self.bentuk_features)
        tab.setLayout(layout)
        return tab

    def create_tekstur_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.tekstur_btn = QPushButton("Ekstrak Tekstur")
        self.tekstur_btn.clicked.connect(self.handle_ekstrak_tekstur)
        self.tekstur_features = QTextEdit()
        self.tekstur_features.setReadOnly(True)
        layout.addWidget(self.tekstur_btn)
        layout.addWidget(QLabel("Fitur Tekstur (GLCM):"))
        layout.addWidget(self.tekstur_features)
        tab.setLayout(layout)
        return tab

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            pixmap = QPixmap(file_name).scaled(300, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.image_path = file_name

    def handle_ekstrak_warna(self):
        if hasattr(self, 'image_path'):
            result = ekstrak_warna(self.image_path)
            teks = f"Mean RGB: {result['mean']}\nStd Dev RGB: {result['std']}"
            self.warna_features.setPlainText(teks)

    def handle_ekstrak_bentuk(self):
        if hasattr(self, 'image_path'):
            result = ekstrak_bentuk(self.image_path)
            teks = f"Luas: {result['area']:.2f}\nKeliling: {result['perimeter']:.2f}"
            self.bentuk_features.setPlainText(teks)

    def handle_ekstrak_tekstur(self):
        if hasattr(self, 'image_path'):
            result = ekstrak_tekstur(self.image_path)
            teks = f"Contrast: {result['contrast']:.2f}\nEntropy: {result['entropy']:.2f}"
            self.tekstur_features.setPlainText(teks)

    def handle_klasifikasi(self):
        if hasattr(self, 'image_path'):
            warna = ekstrak_warna(self.image_path)
            bentuk = ekstrak_bentuk(self.image_path)
            tekstur = ekstrak_tekstur(self.image_path)

            r_mean = warna['mean'][0]
            area = bentuk['area']
            entropy = tekstur['entropy']
            contrast = tekstur['contrast']
            print("Klasifikasi dijalankan")
            print(f"r_mean: {r_mean:.2f}, contrast: {contrast:.2f}, entropy: {entropy:.2f}, area: {area:.2f}")

            # Logika klasifikasi baru
            if area > 30000 and entropy > 4.5:
                kategori = "SISA MAKANAN"
                jenis = "ORGANIK"
            elif entropy < 3 and area > 30000:
                kategori = "BOTOL / PLASTIK"
                jenis = "ANORGANIK"
            elif contrast > 100 and entropy > 4:
                kategori = "KERTAS / KARDUS"
                jenis = "ANORGANIK"
            elif r_mean > 150 and contrast < 10:
                kategori = "PLASTIK CERAH"
                jenis = "ANORGANIK"
            else:
                kategori = "LAINNYA"
                jenis = "TIDAK DIKETAHUI"

            self.result_label.setText(f"Jenis Sampah: {jenis.upper()}")

    def reset_all(self):
        self.image_label.clear()
        self.image_label.setText("Preview Gambar")
        self.warna_features.clear()
        self.bentuk_features.clear()
        self.tekstur_features.clear()
        self.result_label.setText("Kategori: -")
        if hasattr(self, 'image_path'):
            del self.image_path

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SampahClassifierApp()
    window.show()
    sys.exit(app.exec_())
