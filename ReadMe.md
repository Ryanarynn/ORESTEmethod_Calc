Kalkulator ORESTE
Kalkulator ini adalah aplikasi berbasis web untuk Sistem Pendukung Keputusan menggunakan metode ORESTE (Organisation, Rangement et Synthèse de Données Relationnelles). Aplikasi ini memungkinkan pengguna untuk memasukkan alternatif, kriteria, skor performa, dan peringkat kriteria, lalu menghitung urutan alternatif berdasarkan metode ORESTE.
Persyaratan

Python 3.7 atau lebih baru
Library: streamlit, numpy, pandas

Instalasi

Clone atau unduh repositori ini.
Buat virtual environment (opsional):python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Instal library:pip install -r app/requirements.txt


Jalankan aplikasi:streamlit run app/main.py



Cara Penggunaan

Buka aplikasi di browser (biasanya http://localhost:8501).
Masukkan jumlah alternatif dan kriteria.
Isi nama alternatif dan kriteria.
Masukkan skor performa (0-100) untuk setiap alternatif pada setiap kriteria.
Tentukan peringkat kriteria (1 = paling penting).
Klik "Hitung" untuk melihat hasil.
Unduh hasil sebagai file CSV jika diperlukan.

Struktur Folder

app/main.py: Kode utama aplikasi Streamlit.
app/requirements.txt: Daftar library yang diperlukan.
app/assets/: Folder untuk file tambahan (opsional).
README.md: Dokumentasi proyek.

Kontribusi
Silakan buat issue atau pull request untuk saran perbaikan atau fitur tambahan.
Lisensi
MIT License
