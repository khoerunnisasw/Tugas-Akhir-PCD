import os
import sys
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage


class EkstraksiFiturSampah(QWidget):
    def __init__(self):
        super().__init__()
        self.img = None
        self.path = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ekstraksi Fitur Citra Sampah')
        self.labelInput = QLabel('Input Image')
        self.labelOutput = QLabel('Output Image')
        self.labelHasil = QLabel('Hasil Ekstraksi')

        # Set fixed size supaya label gambar tidak terlalu kecil
        self.labelInput.setFixedSize(200, 200)
        self.labelOutput.setFixedSize(200, 200)

        btnBrowse = QPushButton('Pilih Gambar')
        btnWarna = QPushButton('Ekstraksi Warna')
        btnBentuk = QPushButton('Ekstraksi Bentuk')
        btnTekstur = QPushButton('Ekstraksi Tekstur')

        btnBrowse.clicked.connect(self.pilih_gambar)
        btnWarna.clicked.connect(self.ekstrak_warna)
        btnBentuk.clicked.connect(self.ekstrak_bentuk)
        btnTekstur.clicked.connect(self.ekstrak_tekstur)

        layout = QVBoxLayout()
        layout.addWidget(self.labelInput)
        layout.addWidget(self.labelOutput)
        layout.addWidget(self.labelHasil)
        layout.addWidget(btnBrowse)
        layout.addWidget(btnWarna)
        layout.addWidget(btnBentuk)
        layout.addWidget(btnTekstur)

        self.setLayout(layout)

        # Atur ukuran window agar lebih besar dan nyaman
        self.resize(400, 600)

    def pilih_gambar(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '', 'Image Files (*.png *.jpg *.bmp)')
        if path:
            self.path = path
            self.img = cv2.imread(path)
            self.tampilkan_input(path)
            self.labelHasil.setText("Gambar berhasil dimuat. Silakan pilih metode ekstraksi.")

    def tampilkan_input(self, path):
        self.labelInput.setPixmap(QPixmap(path).scaled(self.labelInput.width(), self.labelInput.height()))

    def tampilkan_output(self, img):
        if len(img.shape) == 3:
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qImg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = img.shape
            bytes_per_line = w
            qImg = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        self.labelOutput.setPixmap(QPixmap.fromImage(qImg).scaled(self.labelOutput.width(), self.labelOutput.height()))

    def simpan_dataset(self, kategori, fitur, tipe):
        folder = 'dataset'
        os.makedirs(folder, exist_ok=True)
        df = pd.DataFrame([fitur])
        df['kategori'] = kategori
        csv_path = os.path.join(folder, f'{tipe}.csv')
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

    def klasifikasi_warna(self, fitur):
        return 'Organik' if fitur['h_mean'] < 90 else 'Anorganik'

    def klasifikasi_bentuk(self, fitur):
        return 'Organik' if fitur['aspect_ratio'] < 1.5 else 'Anorganik'

    def klasifikasi_tekstur(self, fitur):
        return 'Organik' if fitur['entropy'] > 4 else 'Anorganik'

    def ekstrak_warna(self):
        if self.img is None:
            self.labelHasil.setText("Silakan pilih gambar terlebih dahulu.")
            return

        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(img_hsv[:, :, 0])
        s_mean = np.mean(img_hsv[:, :, 1])
        v_mean = np.mean(img_hsv[:, :, 2])

        hist_h = cv2.calcHist([img_hsv], [0], None, [8], [0, 180]).flatten()
        hist_s = cv2.calcHist([img_hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([img_hsv], [2], None, [8], [0, 256]).flatten()

        hist = np.concatenate([hist_h, hist_s, hist_v])
        hist = hist / hist.sum()

        fitur = {
            'h_mean': round(h_mean, 2),
            's_mean': round(s_mean, 2),
            'v_mean': round(v_mean, 2),
            **{f'hist_{i}': round(val, 5) for i, val in enumerate(hist)},
            'path': self.path
        }

        kategori = self.klasifikasi_warna(fitur)
        self.simpan_dataset(kategori, fitur, 'warna')
        self.tampilkan_output(self.img)
        self.labelHasil.setText(f"Fitur Warna:\nH Mean: {h_mean:.2f}, S Mean: {s_mean:.2f}, V Mean: {v_mean:.2f}\nKategori: {kategori}")

    def ekstrak_bentuk(self):
        if self.img is None:
            self.labelHasil.setText("Silakan pilih gambar terlebih dahulu.")
            return

        img_copy = self.img.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hu_moments = [0] * 7
        aspect_ratio = 0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            moments = cv2.moments(cnt)
            hu_moments = -np.sign(cv2.HuMoments(moments)).flatten() * np.log10(np.abs(cv2.HuMoments(moments)).flatten() + 1e-10)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0
            cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)

        fitur = {
            **{f'hu_{i}': round(hu, 5) for i, hu in enumerate(hu_moments)},
            'aspect_ratio': round(aspect_ratio, 2),
            'path': self.path
        }

        kategori = self.klasifikasi_bentuk(fitur)
        self.simpan_dataset(kategori, fitur, 'bentuk')
        self.tampilkan_output(img_copy)
        self.labelHasil.setText(f"Fitur Bentuk:\nAspect Ratio: {aspect_ratio:.2f}\nKategori: {kategori}")

    def ekstrak_tekstur(self):
        if self.img is None:
            self.labelHasil.setText("Silakan pilih gambar terlebih dahulu.")
            return

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-6)

        gray_u8 = img_as_ubyte(gray)
        glcm = graycomatrix(gray_u8, [1], [0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        entropy = -np.sum(glcm[:, :, 0, 0] * np.log2(glcm[:, :, 0, 0] + 1e-10))

        fitur = {
            'contrast': round(contrast, 2),
            'entropy': round(entropy, 2),
            **{f'lbp_{i}': round(v, 5) for i, v in enumerate(hist_lbp)},
            'path': self.path
        }

        kategori = self.klasifikasi_tekstur(fitur)
        self.simpan_dataset(kategori, fitur, 'tekstur')
        self.tampilkan_output(gray)
        self.labelHasil.setText(f"Fitur Tekstur:\nContrast: {contrast:.2f}, Entropy: {entropy:.2f}\nKategori: {kategori}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EkstraksiFiturSampah()
    ex.show()
    sys.exit(app.exec_())
