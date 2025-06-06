import os
import cv2
import numpy as np
from ekstraksi_warna import ekstraksi_histogram_warna
from ekstraksi_tekstur import ekstraksi_glcm
from ekstraksi_bentuk import ekstraksi_bentuk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Path dataset
DATASET_PATH = 'dataset'

# Label kategori
LABELS = {
    'plastik': 0,
    'kertas': 1,
    'organik': 2
}

def ekstrak_fitur_lengkap(image_path):
    img = cv2.imread(image_path)
    warna = ekstraksi_histogram_warna(img)
    tekstur = ekstraksi_glcm(img)
    bentuk = ekstraksi_bentuk(img)
    fitur = np.hstack([warna, tekstur, bentuk])
    return fitur

def load_dataset():
    fitur_list = []
    label_list = []

    for kategori, label in LABELS.items():
        folder_path = os.path.join(DATASET_PATH, kategori)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.png')):
                path = os.path.join(folder_path, filename)
                fitur = ekstrak_fitur_lengkap(path)
                fitur_list.append(fitur)
                label_list.append(label)

    return np.array(fitur_list), np.array(label_list)

def train_model():
    X, y = load_dataset()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("=== Hasil Evaluasi Model ===")
    print(classification_report(y_test, y_pred, target_names=LABELS.keys()))
    return knn, scaler

def prediksi_gambar(image_path, model, scaler):
    fitur = ekstrak_fitur_lengkap(image_path)
    fitur = scaler.transform([fitur])
    pred = model.predict(fitur)[0]
    for nama, kode in LABELS.items():
        if kode == pred:
            return nama
    return "Tidak Diketahui"
