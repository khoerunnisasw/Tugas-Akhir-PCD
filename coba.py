import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QGridLayout, QFileDialog,
    QMessageBox, QFrame, QScrollArea
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from skimage.feature import graycomatrix, graycoprops


class SampahClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Sampah - Warna, Bentuk, Tekstur")
        # Meningkatkan ukuran jendela utama untuk memberi ruang lebih
        self.setGeometry(100, 100, 850, 600) # Perbesar lebar dan tinggi
        self.setStyleSheet("background-color: #f5f7fa;")

        # Label untuk menampilkan gambar
        self.label_gambar = QLabel("Belum ada gambar")
        self.label_gambar.setAlignment(Qt.AlignCenter)
        self.label_gambar.setFixedSize(400, 400) # Perbesar sedikit ukuran gambar
        self.label_gambar.setStyleSheet(
            "background-color: white; border: 2px solid #ccc; border-radius: 10px; font-size: 16px; color: #888;"
        )

        # Tombol pilih gambar
        self.btn_pilih = QPushButton("Pilih Gambar")
        self.btn_pilih.setFixedHeight(40)
        self.btn_pilih.setStyleSheet("""
            QPushButton {
                background-color: #007ACC;
                color: white;
                font-weight: bold;
                font-size: 16px;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #005F9E;
            }
        """)
        self.btn_pilih.clicked.connect(self.pilih_gambar)

        # Label fitur dengan scroll area agar bisa scroll jika banyak teks
        self.label_fitur = QLabel("")
        self.label_fitur.setFont(QFont("Consolas", 11))
        self.label_fitur.setStyleSheet("color: #333; padding: 8px;")
        self.label_fitur.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_fitur.setWordWrap(True)

        self.scroll_fitur = QScrollArea()
        self.scroll_fitur.setWidgetResizable(True)
        self.scroll_fitur.setWidget(self.label_fitur)
        self.scroll_fitur.setFixedHeight(220) # Sesuaikan tinggi scroll area
        self.scroll_fitur.setStyleSheet(
            "background: white; border: 1px solid #ccc; border-radius: 8px;"
        )

        # Label hasil klasifikasi
        self.label_hasil = QLabel("Hasil klasifikasi ")
        self.label_hasil.setFont(QFont("Arial", 20, QFont.Bold)) # Perbesar ukuran font
        self.label_hasil.setStyleSheet("color: #004080;")
        self.label_hasil.setAlignment(Qt.AlignCenter)
        self.label_hasil.setWordWrap(True)
        self.label_hasil.setFixedHeight(150) # Perbesar tinggi kotak klasifikasi
        self.label_hasil.setFrameShape(QFrame.Panel)
        self.label_hasil.setFrameShadow(QFrame.Raised)
        self.label_hasil.setLineWidth(2)
        self.label_hasil.setStyleSheet("background-color: #e8f0fe; color: #004080; border-radius: 10px;")


        # Layout grid untuk atur posisi
        layout = QGridLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setHorizontalSpacing(25)
        layout.setVerticalSpacing(15)

        # Atur posisi widget di grid
        # Mengubah row span label_gambar dari 3 menjadi 4 untuk mengakomodasi tinggi baru
        layout.addWidget(self.label_gambar, 0, 0, 4, 1, alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_pilih, 4, 0, alignment=Qt.AlignCenter) # Sesuaikan baris untuk tombol

        layout.addWidget(self.scroll_fitur, 0, 1, 2, 1) # Tetap di baris 0, span 2 baris
        layout.addWidget(self.label_hasil, 2, 1, 3, 1) # Dimulai dari baris 2, span 3 baris untuk memberi ruang lebih

        # Menambahkan stretch untuk baris dan kolom agar layout lebih fleksibel
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 1)
        layout.setRowStretch(3, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        self.setLayout(layout)

    def pilih_gambar(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            try:
                self.tampilkan_gambar(path)
                fitur = self.ekstrak_fitur(path)
                hasil = self.klasifikasi(fitur)
                self.tampilkan_hasil(fitur, hasil)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Gagal memproses gambar:\n{e}")

    def tampilkan_gambar(self, path):
        img = QPixmap(path)
        img = img.scaled(self.label_gambar.width(), self.label_gambar.height(), Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        self.label_gambar.setPixmap(img)
        self.label_gambar.setStyleSheet(
            "background-color: white; border: 2px solid #007ACC; border-radius: 10px;"
        )

    def ekstrak_fitur(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Gambar tidak dapat dibaca")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Warna: rata-rata intensitas merah, hijau, biru
        mean_rgb = np.mean(img_rgb, axis=(0, 1))

        # 2. Bentuk: area kontur terbesar dan circularity
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area = 0
        circularity = 0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)  # nilai 0-1, 1 = lingkaran sempurna

        # 3. Tekstur: GLCM contrast dan entropy
        # Pastikan gambar berwarna abu-abu memiliki kedalaman bit yang sesuai untuk GLCM
        # Misalnya, jika gambar input sudah 8-bit, tidak perlu konversi lagi.
        # Tapi jika Anda menduga ada masalah, bisa tambahkan:
        # gray_8bit = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        # Untuk entropy, perlu penanganan jika ada nilai 0 di glcm agar log tidak menghasilkan -inf
        # Sudah ada 1e-10 di log2, jadi ini cukup.
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

        return {
            'r_mean': mean_rgb[0],
            'g_mean': mean_rgb[1],
            'b_mean': mean_rgb[2],
            'area': area,
            'circularity': circularity,
            'contrast': contrast,
            'entropy': entropy
        }

    def klasifikasi(self, f):
        # Klasifikasi berdasarkan urutan: ORGANIK → PLASTIK → KERTAS

        # ORGANIK: area besar + bentuk tidak teratur (circularity rendah)
        if f['area'] > 10000 and f['circularity'] < 0.2:
            return "ORGANIK (Area besar & bentuk tidak beraturan)"

        # PLASTIK: tekstur tinggi
        elif f['contrast'] > 40 and f['entropy'] > 4:
            return "PLASTIK (Tekstur kompleks)"

        # KERTAS: warna cerah & circularity agak rendah
        elif (f['r_mean'] > 200 and f['g_mean'] > 200 and f['b_mean'] > 200) and f['circularity'] < 0.6:
            return "KERTAS (Warna cerah dan bentuk lembaran)"

        # Tidak dikenali
        else:
            return "TIDAK DIKETAHUI (Fitur tidak sesuai aturan)"

    def tampilkan_hasil(self, fitur, hasil):
        fitur_teks = (
            f"Fitur Ekstraksi:\n"
            f"- R Mean      : {fitur['r_mean']:.2f}\n"
            f"- G Mean      : {fitur['g_mean']:.2f}\n"
            f"- B Mean      : {fitur['b_mean']:.2f}\n"
            f"- Area        : {fitur['area']:.2f}\n"
            f"- Circularity : {fitur['circularity']:.3f}\n"
            f"- Contrast    : {fitur['contrast']:.2f}\n"
            f"- Entropy     : {fitur['entropy']:.2f}"
        )
        self.label_fitur.setText(fitur_teks)
        self.label_hasil.setText(f"Hasil Klasifikasi:\n<b>{hasil}</b>")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SampahClassifier()
    window.show()
    sys.exit(app.exec_())