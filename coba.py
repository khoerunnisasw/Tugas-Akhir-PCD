import os
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# --- Ekstraksi Fitur ---

def ekstrak_hsv_mean(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.mean(hsv)[:3]

def ekstrak_histogram_warna(image, bins=16):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins]*3, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def ekstrak_bentuk(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros(8)
    c = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h != 0 else 0
    return np.append(hu, aspect_ratio)

def ekstrak_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist

def ekstrak_glcm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, correlation, energy, homogeneity])

def ekstraksi_fitur(image):
    fitur_warna = np.concatenate((ekstrak_hsv_mean(image), ekstrak_histogram_warna(image)))
    fitur_bentuk = ekstrak_bentuk(image)
    fitur_lbp = ekstrak_lbp(image)
    fitur_glcm = ekstrak_glcm(image)
    return np.concatenate((fitur_warna, fitur_bentuk, fitur_lbp, fitur_glcm))


# --- Load Dataset ---

def load_dataset(folder_path):
    data, labels, paths = [], [], []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if not os.path.isdir(label_folder):
            continue
        for file in os.listdir(label_folder):
            file_path = os.path.join(label_folder, file)
            img = cv2.imread(file_path)
            if img is None:
                continue
            fitur = ekstraksi_fitur(img)
            data.append(fitur)
            labels.append(label)
            paths.append(file_path)
    return np.array(data), np.array(labels), np.array(paths)


# --- Klasifikasi ---

def klasifikasi_sampah():
    print("Memuat dataset...")
    X, y, paths = load_dataset("dataset")  # Folder: dataset/organik/, dataset/anorganik/

    print("Membagi data latih dan uji...")
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.2, random_state=42
    )

    print("Melatih model SVM...")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    print("\nEvaluasi pada data uji:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("\n=== Prediksi Seluruh Gambar di Dataset ===")
    hasil_klasifikasi = []
    for path, fitur, label_asli in zip(paths, X, y):
        pred = model.predict([fitur])[0]
        nama_file = os.path.basename(path)
        hasil_klasifikasi.append((nama_file, label_asli, pred))
        print(f"File: {nama_file}\tLabel Asli: {label_asli}\tPrediksi: {pred}")

    # Simpan ke CSV
    with open("hasil_klasifikasi.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Nama File", "Label Asli", "Prediksi"])
        writer.writerows(hasil_klasifikasi)
    print("\nHasil klasifikasi disimpan di 'hasil_klasifikasi.csv'")


# --- Main ---

if __name__ == "__main__":
    print("=== Klasifikasi Sampah Berbasis Warna, Bentuk, dan Tekstur ===")
    klasifikasi_sampah()
