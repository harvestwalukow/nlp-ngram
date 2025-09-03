### Struktur Proyek Awal

```
n-gram/
├── app.py              # Aplikasi Flask utama
├── requirements.txt    # Dependensi Python
├── README.md           # Dokumentasi
├── templates/          # Template HTML (Jinja2)
│   └── index.html
└── static/             # Static files (CSS, JS)
    ├── css/
    │   └── style.css
    └── js/
        └── script.js
```

### Prasyarat

- Python 3.8+ terpasang
- Git terpasang (untuk clone/push)

### Cara Menjalankan (dengan Clone Repository)

1. Clone repository:
   ```bash
   git clone <URL_REPOSITORY_GIT_ANDA>
   ```
2. Masuk ke folder proyek:
   ```bash
   cd n-gram
   ```
3. (Opsional) Buat dan aktifkan virtual environment:
   - Windows (PowerShell/CMD):
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
4. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
5. Jalankan aplikasi:
   ```bash
   python app.py
   ```

### Perintah Git yang Umum Digunakan

- Konfigurasi identitas (sekali saja di komputer Anda):

  ```bash
  git config --global user.name "Nama Anda"
  git config --global user.email "email@anda.com"
  ```

- Inisialisasi repository baru (jika memulai dari nol di folder ini):

  ```bash
  git init
  ```

- Menambahkan remote (hubungkan ke repository di GitHub/GitLab):

  ```bash
  git remote add origin <URL_REPOSITORY_GIT_ANDA>
  ```

- Cek status perubahan:

  ```bash
  git status
  ```

- Tambahkan semua perubahan ke staging:

  ```bash
  git add .
  ```

- Buat commit dengan pesan:

  ```bash
  git commit -m "Inisialisasi aplikasi Flask"
  ```

- Push pertama kali ke branch `main` dan set upstream:

  ```bash
  git branch -M main
  git push -u origin main
  ```

- Push perubahan berikutnya:

  ```bash
  git push
  ```

- Tarik perubahan terbaru dari remote:
  ```bash
  git pull
  ```
