# 🧪 Ekstraksi Fitur Citra untuk Klasifikasi Sampah

<<<<<<< HEAD
Proyek ini merupakan implementasi program ekstraksi fitur dari citra RGB untuk keperluan klasifikasi objek sampah. Ekstraksi dilakukan berdasarkan tiga jenis fitur utama: **🎨 warna**, **📐 bentuk**, dan **🧵 tekstur**. Setidaknya terdapat **60 citra RGB** yang digunakan sebagai dataset dengan objek berupa berbagai jenis sampah.
=======
Anggota Kelompok:
- Khoerunnisa Somawijaya (152023096)
- Amanda Pramitha Ramadhani (152023105)
- Shafa Gusti Faradila (152023110)

Proyek ini merupakan implementasi program ekstraksi fitur dari citra RGB untuk keperluan klasifikasi objek sampah. Ekstraksi dilakukan berdasarkan tiga jenis fitur utama: **warna**, **bentuk**, dan **tekstur**. Setidaknya terdapat **60 citra RGB** yang digunakan sebagai dataset dengan objek berupa berbagai jenis sampah.
>>>>>>> 8a42db424301c83dda50e67596933fbcb9c9845f

## 🎯 Tujuan
Mengembangkan tiga program ekstraksi fitur yang dapat mengambil informasi:
- **Warna** dari citra (misalnya untuk kategori kertas)
- **Bentuk** dari citra (misalnya untuk kategori organik)
- **Tekstur** dari citra (misalnya untuk kategori plastik)

> ⚠️ Pembagian kategori sampah dan jenis fitur yang digunakan dapat disesuaikan dengan kebutuhan dan hasil observasi.

## ♻️ Struktur Kategori Sampah
Sebagai contoh, sampah diklasifikasikan ke dalam 3 kategori:
1. **Plastik** – fitur utama: *tekstur*
2. **Kertas** – fitur utama: *warna*
3. **Organik** – fitur utama: *bentuk*

## 🛠️ Komponen Program
Proyek ini terdiri dari tiga modul utama:
- `ekstraksi_warna.py` – untuk mengekstraksi fitur 🎨 warna
- `ekstraksi_bentuk.py` – untuk mengekstraksi fitur 📐 bentuk
- `ekstraksi_tekstur.py` – untuk mengekstraksi fitur 🧵 tekstur

Setiap modul dapat dijalankan secara mandiri dan disesuaikan untuk memproses dataset berdasarkan kategori masing-masing.

## 🗂️ Dataset
- Minimal terdiri dari **60 citra RGB**
- Format citra: `.jpg` atau `.png`
- Dikelompokkan berdasarkan kategori sampah

## 🧰 Teknologi yang Digunakan
<<<<<<< HEAD
- Python
- OpenCV / PIL
- NumPy
- Scikit-image / skimage
=======
- Python 3.x
- [PyQt5](https://pypi.org/project/PyQt5/) – GUI
- [OpenCV](https://opencv.org/) – pengolahan citra
- [NumPy](https://numpy.org/) – representasi data
- [scikit-image](https://scikit-image.org/) – analisis tekstur
- [pandas](https://pandas.pydata.org/) – ekspor data `.csv`
- matplotlib – visualisasi sederhana (opsional)
>>>>>>> 8a42db424301c83dda50e67596933fbcb9c9845f

## ▶️ Cara Menjalankan
1. Pastikan semua dependensi telah terinstall:
    ```bash
    pip install -r requirements.txt
    ```
2. Jalankan salah satu modul ekstraksi:
    ```bash
    python ekstraksi_warna.py
    python ekstraksi_bentuk.py
    python ekstraksi_tekstur.py
    ```

## 📤 Output
<<<<<<< HEAD
Setiap program akan menghasilkan data fitur dalam format `.csv` atau `.npy` yang bisa digunakan untuk tahap klasifikasi atau visualisasi lanjutan.
=======
GUI akan menampilkan hasil ekstraksi dan klasifikasi tiap sampah.
>>>>>>> 8a42db424301c83dda50e67596933fbcb9c9845f

---

💡 *Silakan sesuaikan pembagian kategori dan metode ekstraksi fitur sesuai hasil eksperimen dan kebutuhan proyek.*
