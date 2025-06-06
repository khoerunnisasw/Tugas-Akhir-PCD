import sys
import os
import cv2
import numpy as np
import csv
import pandas as pd
import pickle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QAction,
    QFileDialog, QMessageBox, QVBoxLayout, QWidget, QHBoxLayout,
    QComboBox, QTextEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


class WasteClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Sampah Organik vs Anorganik - Fixed Version")
        self.setGeometry(200, 150, 1000, 700)

        # Inisialisasi model
        self.model = None
        self.scaler = StandardScaler()
        self.is_model_trained = False
        self.dataset_window = None  # Untuk window dataset

        self.setup_ui()
        self.load_existing_model()

    def setup_ui(self):
        # Labels untuk gambar
        self.labelInput = QLabel("Gambar Asli")
        self.labelInput.setAlignment(Qt.AlignCenter)
        self.labelInput.setFixedSize(300, 300)
        self.labelInput.setStyleSheet("border: 1px solid gray;")

        self.labelOutput = QLabel("Hasil Pemrosesan")
        self.labelOutput.setAlignment(Qt.AlignCenter)
        self.labelOutput.setFixedSize(300, 300)
        self.labelOutput.setStyleSheet("border: 1px solid gray;")

        # Layout gambar
        gambarLayout = QHBoxLayout()
        gambarLayout.addWidget(self.labelInput)
        gambarLayout.addWidget(self.labelOutput)

        # Panel kontrol
        controlLayout = QVBoxLayout()

        # Combobox untuk memilih mode
        self.modeCombo = QComboBox()
        self.modeCombo.addItems(["Training Mode", "Prediction Mode"])
        controlLayout.addWidget(QLabel("Mode:"))
        controlLayout.addWidget(self.modeCombo)

        # Combobox untuk label (hanya untuk training mode)
        self.labelCombo = QComboBox()
        self.labelCombo.addItems(["organik", "anorganik"])
        controlLayout.addWidget(QLabel("Label (Training Mode):"))
        controlLayout.addWidget(self.labelCombo)

        # Text area untuk hasil
        self.resultText = QTextEdit()
        self.resultText.setMaximumHeight(200)
        controlLayout.addWidget(QLabel("Hasil:"))
        controlLayout.addWidget(self.resultText)

        # Layout utama
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(gambarLayout)
        mainLayout.addLayout(controlLayout)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

        # Menu bar
        self.setup_menu()

    def setup_menu(self):
        menu = self.menuBar()

        # Menu File
        fileMenu = menu.addMenu("File")

        openAction = QAction("Buka Gambar", self)
        openAction.triggered.connect(self.process_image)
        fileMenu.addAction(openAction)

        # Menu Model
        modelMenu = menu.addMenu("Model")

        trainAction = QAction("Train Model", self)
        trainAction.triggered.connect(self.train_model)
        modelMenu.addAction(trainAction)

        saveModelAction = QAction("Simpan Model", self)
        saveModelAction.triggered.connect(self.save_model)
        modelMenu.addAction(saveModelAction)

        loadModelAction = QAction("Load Model", self)
        loadModelAction.triggered.connect(self.load_model)
        modelMenu.addAction(loadModelAction)

        # Menu Dataset
        datasetMenu = menu.addMenu("Dataset")

        viewDataAction = QAction("Lihat Dataset", self)
        viewDataAction.triggered.connect(self.view_dataset)
        datasetMenu.addAction(viewDataAction)

        clearDataAction = QAction("Hapus Dataset", self)
        clearDataAction.triggered.connect(self.clear_dataset)
        datasetMenu.addAction(clearDataAction)

    def extract_all_features(self, img_path):
        """Ekstrak semua fitur: warna, bentuk, dan tekstur dengan perbaikan"""
        img = cv2.imread(img_path)
        if img is None:
            return None

        features = {}

        # === FITUR WARNA (DIPERBAIKI) ===
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Rata-rata RGB
        features['r_mean'] = np.mean(img_rgb[:, :, 0])
        features['g_mean'] = np.mean(img_rgb[:, :, 1])
        features['b_mean'] = np.mean(img_rgb[:, :, 2])

        # Standar deviasi RGB
        features['r_std'] = np.std(img_rgb[:, :, 0])
        features['g_std'] = np.std(img_rgb[:, :, 1])
        features['b_std'] = np.std(img_rgb[:, :, 2])

        # HSV features
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        features['h_mean'] = np.mean(img_hsv[:, :, 0])
        features['s_mean'] = np.mean(img_hsv[:, :, 1])
        features['v_mean'] = np.mean(img_hsv[:, :, 2])

        # PERBAIKAN: Tambahkan fitur warna yang lebih spesifik
        # Deteksi dominasi warna hijau (untuk daun)
        features['green_dominance'] = features['g_mean'] / (
                    features['r_mean'] + features['g_mean'] + features['b_mean'])

        # Deteksi warna coklat/kuning (untuk daun kering)
        features['brown_yellow_ratio'] = (features['r_mean'] + features['g_mean']) / (features['b_mean'] + 1)

        # Variasi warna (organik cenderung lebih bervariasi)
        features['color_variance'] = np.var([features['r_mean'], features['g_mean'], features['b_mean']])

        # === FITUR BENTUK (DIPERBAIKI) ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perbaikan threshold dengan OTSU
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Noise reduction
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Ambil kontur terbesar
            cnt = max(contours, key=cv2.contourArea)

            # Area dan perimeter
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            features['area'] = area
            features['perimeter'] = perimeter
            features['circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            features['aspect_ratio'] = float(w) / h if h > 0 else 0

            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = area / hull_area if hull_area > 0 else 0

            # PERBAIKAN: Tambahkan fitur bentuk yang lebih baik
            # Extent (rasio area objek terhadap bounding rectangle)
            rect_area = w * h
            features['extent'] = area / rect_area if rect_area > 0 else 0

            # Compactness (ukuran kekompakan bentuk)
            features['compactness'] = (perimeter ** 2) / area if area > 0 else 0

        else:
            features.update({
                'area': 0, 'perimeter': 0, 'circularity': 0,
                'aspect_ratio': 0, 'solidity': 0, 'extent': 0, 'compactness': 0
            })

        # === FITUR TEKSTUR (DIPERBAIKI) ===
        gray_u8 = img_as_ubyte(gray)

        # GLCM features dengan error handling yang lebih baik
        try:
            # Resize jika gambar terlalu besar untuk GLCM
            if gray_u8.shape[0] > 256 or gray_u8.shape[1] > 256:
                gray_u8 = cv2.resize(gray_u8, (256, 256))

            glcm = graycomatrix(gray_u8, [1, 2], [0, 45, 90, 135], levels=256, symmetric=True, normed=True)

            features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
            features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))

            # Entropy
            glcm_normalized = glcm / (np.sum(glcm) + 1e-10)
            features['entropy'] = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))

        except Exception as e:
            print(f"Error in GLCM calculation: {e}")
            features.update({
                'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
                'energy': 0, 'correlation': 0, 'entropy': 0
            })

        # Gabor filter response dengan error handling
        try:
            gabor_responses = []
            for theta in np.arange(0, np.pi, np.pi / 4):
                kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                gabor_responses.append(np.mean(np.abs(filtered)))

            features['gabor_mean'] = np.mean(gabor_responses)
            features['gabor_std'] = np.std(gabor_responses)

        except Exception as e:
            print(f"Error in Gabor filter: {e}")
            features['gabor_mean'] = 0
            features['gabor_std'] = 0

        # PERBAIKAN: Tambahkan fitur statistik tambahan
        features['brightness'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)

        return features

    def display_image(self, img_path, label_widget):
        """Display image in label widget"""
        try:
            pixmap = QPixmap(img_path)
            scaled_pixmap = pixmap.scaled(label_widget.width(), label_widget.height(),
                                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error displaying image: {e}")

    def display_cv_image(self, cv_img, label_widget):
        """Display OpenCV image in label widget"""
        try:
            if len(cv_img.shape) == 3:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                h, w = cv_img.shape
                bytes_per_line = w
                qt_image = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(label_widget.width(), label_widget.height(),
                                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error displaying CV image: {e}")

    def process_image(self):
        """Proses gambar untuk training atau prediksi"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_path:
            return

        # Tampilkan gambar input
        self.display_image(file_path, self.labelInput)

        # Ekstrak fitur
        features = self.extract_all_features(file_path)
        if features is None:
            QMessageBox.warning(self, "Error", "Gagal memproses gambar!")
            return

        mode = self.modeCombo.currentText()

        if mode == "Training Mode":
            self.add_to_dataset(features, file_path)
        else:
            self.predict_image(features, file_path)

    def add_to_dataset(self, features, img_path):
        """Tambahkan data ke dataset untuk training"""
        label = self.labelCombo.currentText()

        # Simpan ke CSV
        csv_file = 'dataset/waste_features.csv'
        os.makedirs('dataset', exist_ok=True)

        # Tambahkan path dan label
        features['image_path'] = img_path
        features['label'] = label

        # Tulis ke CSV
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=features.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(features)

        # Tampilkan hasil
        result_text = f"Data berhasil ditambahkan ke dataset!\n"
        result_text += f"Label: {label}\n"
        result_text += f"Jumlah fitur: {len(features) - 2}\n"  # -2 untuk path dan label
        result_text += f"File: {os.path.basename(img_path)}\n\n"

        # Tampilkan beberapa fitur penting
        result_text += f"Fitur Warna:\n"
        result_text += f"- RGB: ({features['r_mean']:.1f}, {features['g_mean']:.1f}, {features['b_mean']:.1f})\n"
        result_text += f"- HSV: ({features['h_mean']:.1f}, {features['s_mean']:.1f}, {features['v_mean']:.1f})\n"
        result_text += f"- Green Dominance: {features['green_dominance']:.3f}\n\n"
        result_text += f"Fitur Bentuk:\n"
        result_text += f"- Area: {features['area']:.0f}\n"
        result_text += f"- Circularity: {features['circularity']:.3f}\n"

        self.resultText.setText(result_text)

        # Tampilkan gambar yang diproses
        img = cv2.imread(img_path)
        self.display_cv_image(img, self.labelOutput)

    def predict_image(self, features, img_path):
        """Prediksi label gambar"""
        if not self.is_model_trained:
            QMessageBox.warning(self, "Warning", "Model belum di-training! Silakan train model terlebih dahulu.")
            return

        # Konversi features ke array (hilangkan path jika ada)
        feature_dict = {k: v for k, v in features.items() if k not in ['image_path', 'label']}
        feature_names = list(feature_dict.keys())
        feature_values = list(feature_dict.values())

        # Prediksi
        try:
            X = np.array(feature_values).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]

            # Cari index untuk organik dan anorganik
            organik_idx = list(self.model.classes_).index('organik') if 'organik' in self.model.classes_ else 0
            anorganik_idx = list(self.model.classes_).index('anorganik') if 'anorganik' in self.model.classes_ else 1

            # Tampilkan hasil
            result_text = f"🔍 HASIL PREDIKSI\n"
            result_text += f"==================\n"
            result_text += f"Label: {prediction.upper()}\n"
            result_text += f"Confidence: {max(probability):.1%}\n\n"
            result_text += f"Probabilitas Detail:\n"
            result_text += f"🌱 Organik: {probability[organik_idx]:.1%}\n"
            result_text += f"🗂️ Anorganik: {probability[anorganik_idx]:.1%}\n\n"

            # Tampilkan fitur kunci
            result_text += f"Fitur Kunci:\n"
            result_text += f"- Green Dominance: {features['green_dominance']:.3f}\n"
            result_text += f"- Circularity: {features['circularity']:.3f}\n"
            result_text += f"- Contrast: {features['contrast']:.2f}\n\n"

            result_text += f"File: {os.path.basename(img_path)}"

            self.resultText.setText(result_text)

            # Tampilkan gambar dengan hasil
            img = cv2.imread(img_path)
            # Tambahkan text hasil pada gambar
            color = (0, 255, 0) if prediction == 'organik' else (0, 0, 255)
            cv2.putText(img, f"{prediction.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img, f"{max(probability):.1%}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            self.display_cv_image(img, self.labelOutput)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat prediksi: {str(e)}")
            print(f"Prediction error details: {e}")

    def train_model(self):
        """Train model machine learning"""
        csv_file = 'dataset/waste_features.csv'

        if not os.path.exists(csv_file):
            QMessageBox.warning(self, "Warning",
                                "Dataset tidak ditemukan! Silakan tambahkan data training terlebih dahulu.")
            return

        try:
            # Load dataset
            df = pd.read_csv(csv_file)

            if len(df) < 2:
                QMessageBox.warning(self, "Warning", "Dataset terlalu sedikit! Minimal 2 data diperlukan.")
                return

            # Pisahkan features dan labels
            X = df.drop(['image_path', 'label'], axis=1, errors='ignore')
            y = df['label']

            # Hapus kolom dengan nilai NaN atau infinite
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)

            # Split data
            if len(df) > 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = X, X, y, y

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model dengan parameter yang lebih baik
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # Menangani ketidakseimbangan kelas
            )
            self.model.fit(X_train_scaled, y_train)

            # Evaluasi
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Tampilkan hasil
            result_text = f"🎯 MODEL TRAINING SELESAI!\n"
            result_text += f"================================\n"
            result_text += f"Dataset: {len(df)} sampel\n"
            result_text += f"Features: {len(X.columns)}\n"
            result_text += f"Training Accuracy: {train_acc:.1%}\n"
            result_text += f"Testing Accuracy: {test_acc:.1%}\n\n"
            result_text += f"Distribusi Label:\n"
            for label, count in y.value_counts().items():
                result_text += f"- {label}: {count} sampel\n"

            result_text += f"\nTop 5 Fitur Penting:\n"
            for i, (_, row) in enumerate(feature_importance.head().iterrows()):
                result_text += f"{i + 1}. {row['feature']}: {row['importance']:.3f}\n"

            self.resultText.setText(result_text)
            self.is_model_trained = True

            QMessageBox.information(self, "Success", "Model berhasil di-training!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat training: {str(e)}")
            print(f"Training error details: {e}")

    def save_model(self):
        """Simpan model yang sudah di-training"""
        if not self.is_model_trained:
            QMessageBox.warning(self, "Warning", "Tidak ada model untuk disimpan!")
            return

        try:
            os.makedirs('models', exist_ok=True)

            # Simpan model dan scaler
            with open('models/waste_classifier.pkl', 'wb') as f:
                pickle.dump(self.model, f)

            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            QMessageBox.information(self, "Success", "Model berhasil disimpan!")
            self.resultText.setText("Model berhasil disimpan ke folder 'models'!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat menyimpan model: {str(e)}")

    def load_model(self):
        """Load model yang sudah disimpan"""
        try:
            if os.path.exists('models/waste_classifier.pkl') and os.path.exists('models/scaler.pkl'):
                with open('models/waste_classifier.pkl', 'rb') as f:
                    self.model = pickle.load(f)

                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)

                self.is_model_trained = True
                QMessageBox.information(self, "Success", "Model berhasil di-load!")

                result_text = "✅ Model berhasil di-load!\n"
                result_text += "Sistem siap untuk prediksi.\n\n"
                result_text += "Cara penggunaan:\n"
                result_text += "1. Pilih 'Prediction Mode'\n"
                result_text += "2. Klik 'Buka Gambar' untuk prediksi"

                self.resultText.setText(result_text)
            else:
                QMessageBox.warning(self, "Warning", "File model tidak ditemukan!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat load model: {str(e)}")

    def load_existing_model(self):
        """Load model yang ada saat startup"""
        try:
            if os.path.exists('models/waste_classifier.pkl') and os.path.exists('models/scaler.pkl'):
                with open('models/waste_classifier.pkl', 'rb') as f:
                    self.model = pickle.load(f)

                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)

                self.is_model_trained = True
                self.resultText.setText("Model tersedia. Siap untuk prediksi!")
        except:
            self.resultText.setText("Belum ada model. Silakan training terlebih dahulu.")

    def view_dataset(self):
        """Lihat isi dataset"""
        csv_file = 'dataset/waste_features.csv'

        if not os.path.exists(csv_file):
            QMessageBox.information(self, "Info", "Dataset belum ada!")
            return

        try:
            df = pd.read_csv(csv_file)

            # Create a new window to display the dataset
            if self.dataset_window is not None:
                self.dataset_window.close()

            self.dataset_window = QWidget()
            self.dataset_window.setWindowTitle(f"Dataset - {len(df)} sampel")
            self.dataset_window.setGeometry(200, 150, 1000, 600)

            # Create a table to display the dataset
            table = QTableWidget()
            table.setRowCount(len(df))
            table.setColumnCount(len(df.columns))

            # Set the column headers
            table.setHorizontalHeaderLabels(df.columns)

            # Fill the table with data
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    value = df.iat[i, j]
                    if isinstance(value, float):
                        table.setItem(i, j, QTableWidgetItem(f"{value:.3f}"))
                    else:
                        table.setItem(i, j, QTableWidgetItem(str(value)))

            # Layout for the dataset window
            layout = QVBoxLayout()

            # Add summary label
            summary_label = QLabel(
                f"Total: {len(df)} sampel | Organik: {len(df[df['label'] == 'organik'])} | Anorganik: {len(df[df['label'] == 'anorganik'])}")
            layout.addWidget(summary_label)
            layout.addWidget(table)

            self.dataset_window.setLayout(layout)
            self.dataset_window.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saat membaca dataset: {str(e)}")

    def clear_dataset(self):
        """Hapus dataset"""
        reply = QMessageBox.question(self, "Konfirmasi",
                                     "Apakah Anda yakin ingin menghapus semua data dataset?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            csv_file = 'dataset/waste_features.csv'
            if os.path.exists(csv_file):
                os.remove(csv_file)
                QMessageBox.information(self, "Success", "Dataset berhasil dihapus!")
                self.resultText.setText("Dataset telah dihapus. Mulai dari awal.")
            else:
                QMessageBox.information(self, "Info", "Dataset sudah kosong!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WasteClassifier()
    window.show()
    sys.exit(app.exec_())