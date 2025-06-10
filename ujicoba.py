import sys
import os
import cv2
import numpy as np
import csv
import mahotas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QAction,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class FiturExtractor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ekstraksi Fitur Sampah")
        self.setGeometry(200, 150, 800, 600)

        self.labelInput = QLabel("Gambar Asli")
        self.labelInput.setAlignment(Qt.AlignCenter)
        self.labelInput.setFixedSize(300, 300)

        self.labelOutput = QLabel("Hasil Ekstraksi")
        self.labelOutput.setAlignment(Qt.AlignCenter)
        self.labelOutput.setFixedSize(300, 300)

        self.labelHasil = QLabel("")
        self.labelHasil.setAlignment(Qt.AlignTop)
        self.labelHasil.setWordWrap(True)

        gambarLayout = QHBoxLayout()
        gambarLayout.addWidget(self.labelInput)
        gambarLayout.addWidget(self.labelOutput)

        layout = QVBoxLayout()
        layout.addLayout(gambarLayout)
        layout.addWidget(self.labelHasil)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        menu = self.menuBar()
        menuEkstraksi = menu.addMenu("Ekstraksi Fitur")

        self.actionWarna = QAction("Ekstraksi Warna", self)
        self.actionWarna.triggered.connect(self.ekstrak_warna)

        self.actionBentuk = QAction("Ekstraksi Bentuk", self)
        self.actionBentuk.triggered.connect(self.ekstrak_bentuk)

        self.actionTekstur = QAction("Ekstraksi Tekstur", self)
        self.actionTekstur.triggered.connect(self.ekstrak_tekstur)

        menuEkstraksi.addAction(self.actionWarna)
        menuEkstraksi.addAction(self.actionBentuk)
        menuEkstraksi.addAction(self.actionTekstur)

    def buka_gambar(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        return file_path if file_path else None

    def tampilkan_input(self, path):
        image = QPixmap(path).scaled(self.labelInput.width(), self.labelInput.height(), Qt.KeepAspectRatio)
        self.labelInput.setPixmap(image)

    def tampilkan_output(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.labelOutput.width(), self.labelOutput.height(), Qt.KeepAspectRatio)
        self.labelOutput.setPixmap(pixmap)

    def simpan_dataset(self, kategori, fitur, tipe_fitur):
        os.makedirs(f'dataset/{kategori}', exist_ok=True)
        filename = os.path.basename(fitur['path'])
        save_path = f'dataset/{kategori}/{tipe_fitur}_{filename}'
        cv2.imwrite(save_path, fitur['img'])

        csv_file = f'dataset/fitur_{tipe_fitur}.csv'
        header = [k for k in fitur if k not in ['img', 'path']] + ['kategori']
        row = [fitur[k] for k in header if k != 'kategori'] + [kategori]

        write_header = not os.path.exists(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    def apply_gabor_filter(self, gray):
        gabor_output = np.zeros_like(gray, dtype=np.float32)
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            gabor_output = np.maximum(gabor_output, filtered)
        gabor_output = cv2.normalize(gabor_output, None, 0, 255, cv2.NORM_MINMAX)
        return gabor_output.astype(np.uint8)

    def ekstrak_warna(self):
        path = self.buka_gambar()
        if path:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img_rgb.shape
            img_reshaped = img_rgb.reshape((-1, 3)).astype(np.float32)

            # Terapkan KMeans clustering untuk segmentasi warna
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 3
            _, labels, centers = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((img_rgb.shape)).astype(np.uint8)

            # Rata-rata RGB
            r_mean = np.mean(img_rgb[:, :, 0])
            g_mean = np.mean(img_rgb[:, :, 1])
            b_mean = np.mean(img_rgb[:, :, 2])

            # Konversi ke HSV
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            h_mean = np.mean(img_hsv[:, :, 0])
            s_mean = np.mean(img_hsv[:, :, 1])
            v_mean = np.mean(img_hsv[:, :, 2])

            # Konversi ke CMYK (manual)
            r_norm, g_norm, b_norm = r_mean / 255.0, g_mean / 255.0, b_mean / 255.0
            k = 1 - max(r_norm, g_norm, b_norm)
            if k < 1:
                c = (1 - r_norm - k) / (1 - k)
                m = (1 - g_norm - k) / (1 - k)
                y = (1 - b_norm - k) / (1 - k)
            else:
                c = m = y = 0

            # Deteksi kondisi pencahayaan berdasarkan nilai Value (V)
            if v_mean > 180:
                kondisi = "Terang"
            elif v_mean > 100:
                kondisi = "Redup"
            else:
                kondisi = "Berembun"

            # Konversi segmented image ke grayscale lalu colormap
            gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
            thermal_colored = cv2.applyColorMap(gray_segmented, cv2.COLORMAP_JET)

            fitur = {
                'r_mean': round(r_mean, 2),
                'g_mean': round(g_mean, 2),
                'b_mean': round(b_mean, 2),
                'h_mean': round(h_mean, 2),
                's_mean': round(s_mean, 2),
                'v_mean': round(v_mean, 2),
                'c_cmyk': round(c, 2),
                'm_cmyk': round(m, 2),
                'y_cmyk': round(y, 2),
                'k_cmyk': round(k, 2),
                'kondisi': kondisi,
                'path': path,
                'img': thermal_colored  # Gambar dengan tampilan thermal
            }

            kategori = self.klasifikasi_warna(fitur)
            self.simpan_dataset(kategori, fitur, 'warna')

            self.tampilkan_input(path)
            self.tampilkan_output(fitur['img'])

            self.labelHasil.setText(
                f"Fitur Warna:\n"
                f"R Mean: {r_mean:.2f}, G Mean: {g_mean:.2f}, B Mean: {b_mean:.2f}\n"
                f"H Mean: {h_mean:.2f}, S Mean: {s_mean:.2f}, V Mean: {v_mean:.2f}\n"
                f"CMYK: C={c:.2f}, M={m:.2f}, Y={y:.2f}, K={k:.2f}\n"
                f"Kondisi Citra: {kondisi}\n"
                f"Kategori: {kategori}"
            )

    def ekstrak_bentuk(self):
        path = self.buka_gambar()
        if path:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            area, circularity = 0, 0
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Deteksi kondisi pencahayaan menggunakan komponen Value (V)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            v_mean = np.mean(img_hsv[:, :, 2])

            if v_mean > 180:
                kondisi = "Terang"
            elif v_mean > 100:
                kondisi = "Redup"
            else:
                kondisi = "Berembun"

            fitur = {
                'area': round(area, 2),
                'circularity': round(circularity, 3),
                'v_mean': round(v_mean, 2),
                'kondisi': kondisi,
                'path': path,
                'img': img
            }

            kategori = self.klasifikasi_bentuk(fitur)
            self.simpan_dataset(kategori, fitur, 'bentuk')

            self.tampilkan_input(path)
            self.tampilkan_output(thresh if contours else img)

            self.labelHasil.setText(
                f"Fitur Bentuk:\n"
                f"Area: {area:.2f}, Circularity: {circularity:.3f}\n"
                f"Kondisi Citra: {kondisi}\n"
                f"Kategori: {kategori}"
            )

    def ekstrak_tekstur(self):
        path = self.buka_gambar()
        if path:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Ekstrak fitur Haralick dari mahotas
            haralick_features = mahotas.features.haralick(gray).mean(axis=0)  # rata-rata dari semua arah

            contrast = haralick_features[1]  # biasanya indeks 1 adalah contrast
            entropy = haralick_features[8]  # biasanya indeks 8 adalah entropy

            # Deteksi kondisi pencahayaan
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            v_mean = np.mean(img_hsv[:, :, 2])

            if v_mean > 180:
                kondisi = "Terang"
            elif v_mean > 100:
                kondisi = "Redup"
            else:
                kondisi = "Berembun"

            fitur = {
                'contrast': round(contrast, 2),
                'entropy': round(entropy, 2),
                'v_mean': round(v_mean, 2),
                'kondisi': kondisi,
                'path': path,
                'img': img
            }

            kategori = self.klasifikasi_tekstur(fitur)
            self.simpan_dataset(kategori, fitur, 'tekstur')

            self.tampilkan_input(path)
            gabor_result = self.apply_gabor_filter(gray)
            self.tampilkan_output(gabor_result)

            self.labelHasil.setText(
                f"Fitur Tekstur:\n"
                f"Contrast: {fitur['contrast']}, Entropy: {fitur['entropy']}\n"
                f"Kondisi Citra: {kondisi}\n"
                f"Kategori: {kategori}"
            )

    def klasifikasi_warna(self, f):
        """
        Klasifikasi berdasarkan karakteristik warna yang lebih akurat
        """
        r, g, b = f['r_mean'], f['g_mean'], f['b_mean']
        h, s, v = f['h_mean'], f['s_mean'], f['v_mean']

        # 1. Deteksi warna hijau (daun hidup) - Hue sekitar 60-180 dengan saturasi tinggi
        if 60 <= h <= 180 and s > 50:
            return "organik"

        # 2. Deteksi warna coklat/kuning (daun kering) - Hue 10-60 atau tinggi
        if (10 <= h <= 60 or h >= 300) and s > 30:
            return "organik"

        # 3. Deteksi warna merah/oranye (daun musim gugur) - seperti gambar Anda
        if (h <= 30 or h >= 330) and s > 40 and v > 100:
            return "organik"

        # 4. Cek dominasi warna natural organik
        # Organik biasanya memiliki komponen hijau atau kuning yang signifikan
        if g > r * 0.8 or (r > 120 and g > 80):  # Kombinasi merah-hijau untuk daun
            return "organik"

        # 5. Material anorganik biasanya memiliki warna yang sangat kontras
        # atau sangat abu-abu (R≈G≈B)
        rgb_variance = np.var([r, g, b])
        if rgb_variance < 100 and v > 150:  # Abu-abu terang (plastik, logam)
            return "anorganik"

        # 6. Warna sangat cerah dan saturasi rendah = kemungkinan anorganik
        if v > 200 and s < 30:
            return "anorganik"

        # Default: jika tidak memenuhi kriteria anorganik, anggap organik
        return "organik"

    def klasifikasi_bentuk(self, f):
        """
        Perbaikan klasifikasi bentuk
        """
        # Organik: bentuk tidak beraturan (circularity rendah) dan area bervariasi
        # Anorganik: bentuk lebih beraturan (botol, kaleng, dll)

        if f['circularity'] < 0.3:  # Bentuk sangat tidak beraturan = organik
            return "organik"
        elif f['circularity'] > 0.7 and f['area'] > 5000:  # Bulat + besar = anorganik
            return "anorganik"
        elif f['area'] < 1000:  # Terlalu kecil, kemungkinan noise
            return "organik"  # Default ke organik untuk debris kecil
        else:
            return "organik"  # Default

    def klasifikasi_tekstur(self, f):
        """
        Perbaikan klasifikasi tekstur
        """
        # Organik: tekstur lebih kompleks dan bervariasi
        # Anorganik: tekstur lebih seragam

        if f['contrast'] > 8 and f['entropy'] > 4:  # Tekstur sangat kompleks
            return "organik"
        elif f['contrast'] < 3 and f['entropy'] < 2:  # Tekstur sangat halus
            return "anorganik"
        else:
            # Gunakan kombinasi contrast dan entropy
            texture_score = f['contrast'] * 0.6 + f['entropy'] * 0.4
            return "organik" if texture_score > 4 else "anorganik"

    def klasifikasi_final(self, f):
        """
        Sistem voting dengan bobot yang lebih seimbang
        """
        warna = self.klasifikasi_warna(f)
        bentuk = self.klasifikasi_bentuk(f)
        tekstur = self.klasifikasi_tekstur(f)

        print("Debug Info:")
        print("Fitur:", f)
        print("Warna:", warna)
        print("Bentuk:", bentuk)
        print("Tekstur:", tekstur)

        # Voting dengan prioritas pada warna (paling reliable untuk sampah)
        votes = [warna, bentuk, tekstur]
        organik_count = votes.count("organik")

        # Jika warna mendeteksi organik dan minimal 1 fitur lain setuju
        if warna == "organik" and organik_count >= 2:
            return "organik"
        # Jika mayoritas mengatakan organik
        elif organik_count >= 2:
            return "organik"
        else:
            return "anorganik"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FiturExtractor()
    window.show()
    sys.exit(app.exec_())