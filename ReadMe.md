# 🧮 ORESTE Decision Support Calculator

**ORESTE Calculator** adalah aplikasi berbasis web untuk *Sistem Pendukung Keputusan (SPK)* yang menggunakan metode **ORESTE** (Organisation, Rangement et Synthèse de Données Relationnelles). Aplikasi ini membantu pengguna dalam menentukan urutan peringkat alternatif berdasarkan nilai performa dan tingkat kepentingan kriteria.

---

## 🚀 Fitur Utama

- Input jumlah dan nama **alternatif** serta **kriteria**
- Penilaian performa alternatif (0–100) terhadap setiap kriteria
- Penentuan peringkat kepentingan kriteria
- Proses perhitungan dan penentuan urutan alternatif secara otomatis
- Tampilan hasil lengkap dan interaktif
- Fitur **unduh hasil** dalam format CSV

---

## 📦 Persyaratan

- Python **3.7** atau lebih baru
- Library Python:
  - `streamlit`
  - `numpy`
  - `pandas`

---

## 🛠️ Instalasi

1. **Clone atau unduh repositori:**

   ```bash
   git clone https://github.com/username/oreste-calculator.git
   cd oreste-calculator
   ```

2. **(Opsional) Buat virtual environment:**

   ```bash
   python -m venv venv
   ```

   Aktifkan environment:

   - **Linux/macOS**:

     ```bash
     source venv/bin/activate
     ```

   - **Windows**:

     ```bash
     venv\Scripts\activate
     ```

3. **Install library yang dibutuhkan:**

   ```bash
   pip install -r app/requirements.txt
   ```

4. **Jalankan aplikasi:**

   ```bash
   streamlit run app/main.py
   ```

---

## 🌐 Cara Penggunaan

1. Akses aplikasi di browser Anda: [http://localhost:8501](http://localhost:8501)
2. Masukkan jumlah **alternatif** dan **kriteria**
3. Isi nama alternatif dan nama kriteria
4. Masukkan skor performa (0–100) untuk setiap alternatif terhadap setiap kriteria
5. Tentukan **peringkat kepentingan** untuk tiap kriteria (1 = paling penting)
6. Klik tombol **"Hitung"**
7. Lihat hasil perhitungan dan peringkat alternatif
8. (Opsional) Unduh hasil dalam format CSV

---

## 📁 Struktur Folder

```
app/
│
├── main.py              # Kode utama aplikasi Streamlit
├── requirements.txt     # Daftar dependensi
├── assets/              # (Opsional) File tambahan seperti gambar/logo
│
README.md                # Dokumentasi proyek
```

---

## 🤝 Kontribusi

Kontribusi sangat terbuka! Anda dapat:

- Membuat issue untuk pelaporan bug atau permintaan fitur
- Mengajukan pull request untuk perbaikan atau penambahan fitur

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

## 🧠 Tentang Metode ORESTE

ORESTE adalah metode *multi-criteria decision analysis* (MCDA) yang menggunakan pendekatan peringkat ordinal dan relasional untuk mengevaluasi alternatif berdasarkan sejumlah kriteria. Cocok digunakan ketika data bersifat kualitatif dan sulit dikonversi menjadi nilai kuantitatif secara langsung.
