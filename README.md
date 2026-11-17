# Analisis Distribusi Makan Bergizi Gratis (MBG) 2026 dengan PySpark

Tugas mata kuliah Teknologi Data. Program ini menganalisis 13.000 data distribusi paket MBG di 38 provinsi menggunakan Apache Spark (PySpark).

**Rumusan masalah:**
1. Faktor logistik apa yang berhubungan dengan sisa makanan (*plate waste*)?
2. Wilayah mana yang paling bermasalah?

*Plate waste* adalah persentase makanan yang sudah disajikan tetapi tidak dimakan. Rata-ratanya di data ini 13,34% per paket.

## Cara menjalankan

### Persyaratan
| Kebutuhan | Versi yang dipakai | Cek dengan |
|---|---|---|
| Python | 3.14 | `python3 --version` |
| Java (dibutuhkan Spark) | 17 | `java -version` |

### Langkah
```bash
# sekali saja: buat environment dan pasang library
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# setiap kali menjalankan
source .venv/bin/activate
python main.py
```
Hasil analisis tampil di terminal, dan grafik feature importance disimpan ke `outputs/feature_importance.png`.

### Struktur proyek
```
.
├── main.py              program utama 
├── requirements.txt     daftar library
├── data/
│   ├── indonesia_mbg_nationwide_enriched_2026.csv   data utama (17,46 MB, 13.000 baris × 26 kolom)
│   └── provinsi_pulau.csv                           lookup 38 provinsi → pulau & zona waktu
└── outputs/
    └── feature_importance.png
```

---

## Hasil

### 1. Faktor yang berhubungan dengan plate waste (MLlib, feature importance)

| # | Faktor | Kelompok | Importance |
|---|---|---|---|
| 1 | Kalori | menu | 0,429 |
| 2 | Suhu saat tiba | logistik | 0,181 |
| 3 | Protein (gram) | menu | 0,153 |
| 4 | Waktu transit | logistik | 0,067 |
| 5 | Total biaya per porsi | biaya | 0,048 |
| 6 | Jarak pengiriman | logistik | 0,044 |
| 7 | Jenis protein | menu | 0,039 |
| 8 | Moda transportasi | logistik | 0,027 |
| 9 | Klaster wilayah | wilayah | 0,012 |

### 2. Persentase non-compliant per kelompok pulau (SQL)

| Kelompok pulau | % non-compliant | Rata-rata transit | % tiba suhu aman |
|---|---|---|---|
| Maluku dan Papua | 17,00% | 41,2 menit | 54,7% |
| Bali dan Nusa Tenggara | 9,75% | 31,3 menit | 67,8% |
| Kalimantan | 8,25% | 30,1 menit | 68,6% |
| Sumatera | 7,17% | 26,2 menit | 77,0% |
| Sulawesi | 6,69% | 26,4 menit | 77,6% |
| Jawa | 2,70% | 19,4 menit | 84,1% |

### 3. Top 10 provinsi dengan plate waste tertinggi (SQL)

Papua (14,57%), Sulawesi Barat (14,53%), Papua Pegunungan (14,35%), Papua Selatan (14,33%), Kalimantan Utara (14,31%), Bengkulu (14,11%), Nusa Tenggara Timur (14,07%), Papua Barat Daya (13,97%), Bangka Belitung (13,95%), Papua Tengah (13,92%).

### 4. Provinsi dengan plate waste tertinggi per klaster (SQL, window function)

| Klaster | Peringkat 1 | Plate waste |
|---|---|---|
| Wilayah 3T Perbatasan & Pedalaman | Kepulauan Riau | 16,73% |
| Luar Jawa Urban | Kalimantan Barat | 15,06% |
| Daerah Pedesaan & Kepulauan | Sulawesi Barat | 14,80% |
| Jawa Metropolitan | Banten | 13,24% |

### 5. Biaya per tier kepatuhan (SQL)

| Tier | Rata-rata total biaya | Rata-rata plate waste |
|---|---|---|
| Exemplary-Gold-Standard | Rp13.639 | 6,72% |
| Standard-Compliant | Rp13.320 | 12,77% |
| Marginal-Quality-At-Risk | Rp13.890 | 17,67% |
| Non-Compliant-Intervention-Required | Rp14.435 | 18,88% |

## Kesimpulan

1. **Faktor logistik terpenting adalah suhu makanan saat tiba**, diikuti waktu transit dan jarak. Faktor terbesar secara keseluruhan berasal dari menu (kalori dan protein).
2. **Wilayah paling bermasalah adalah Maluku dan Papua**: non-compliant 17% (Jawa 2,7%), waktu transit 2 kali lebih lama, dan hanya 54,7% paket tiba dalam suhu aman.
3. **Biaya bukan penyebab utama**: distribusi non-compliant justru memiliki biaya per porsi tertinggi.

## Penggunaan AI

Proyek ini dikerjakan dengan bantuan AI ([Claude](https://claude.ai) dari Anthropic):

- **Perancangan analisis** (pemilihan topik, rumusan masalah, pengecekan data, tabel lookup provinsi, cara setup environment dan referensi script python) menggunakan Claude. Percakapannya dapat dilihat di: https://claude.ai/share/fafee8c3-2a22-4901-981d-e1ee7c8b17f6

