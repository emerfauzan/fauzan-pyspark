"""
Analisis Distribusi Makan Bergizi Gratis (MBG) 2026 dengan PySpark

Rumusan masalah:
    Faktor logistik apa yang berhubungan dengan sisa makanan (plate waste),
    dan wilayah mana yang paling bermasalah?

Alur program (sesuai komponen penilaian):
    1. Dataset             -> baca CSV, lihat ukuran data
    2. Tampilan DataFrame  -> skema, contoh baris, statistik
    3. Operasi row         -> buang data kosong, duplikat, dan nilai tidak wajar
    4. Operasi column      -> tambah kolom baru, buang kolom, join tabel pulau
    5. Fitur Spark: SQL    -> query ringkasan per wilayah
    6. Fitur Spark: MLlib  -> model Random Forest untuk mencari faktor terpenting
    7. Solusi              -> grafik untuk laporan (folder outputs/)

Cara menjalankan:
    source .venv/bin/activate
    python main.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # simpan grafik ke file tanpa membuka jendela
import matplotlib.pyplot as plt

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Lokasi file, dihitung dari folder tempat main.py berada
FOLDER = Path(__file__).parent
DATA_PATH = FOLDER / "data" / "indonesia_mbg_nationwide_enriched_2026.csv"
LOOKUP_PATH = FOLDER / "data" / "provinsi_pulau.csv"
OUTPUT_FOLDER = FOLDER / "outputs"


def title(teks):
    """Cetak title bagian supaya output di terminal mudah dibaca."""
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# =============================================================================
# 0. Menyalakan Spark
# =============================================================================
def init_spark():
    """Membuat SparkSession, yaitu 'pintu masuk' untuk semua fitur Spark."""
    spark = (
        SparkSession.builder
        .appName("Analisis-MBG")
        .master("local[*]")                          # jalan di laptop, pakai semua core CPU
        .config("spark.sql.shuffle.partitions", "8")  # data kecil, cukup 8 partisi
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")           # sembunyikan log yang tidak penting
    return spark


# =============================================================================
# 1. Dataset
# =============================================================================
def read_data(spark):
    """Membaca CSV utama dan tabel lookup provinsi -> pulau."""
    df = (
        spark.read
        .option("header", True)       # baris pertama berisi nama kolom
        .option("inferSchema", True)  # Spark menebak tipe kolom (angka/teks)
        .option("escape", '"')        # kolom teks panjang berisi tanda kutip
        .csv(str(DATA_PATH))
    )
    lookup = spark.read.option("header", True).csv(str(LOOKUP_PATH))

    size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
    print(f"Ukuran file : {size_mb:.2f} MB")
    print(f"Jumlah baris: {df.count():,}")
    print(f"Jumlah kolom: {len(df.columns)}")
    return df, lookup


# =============================================================================
# 2. Tampilan DataFrame
# =============================================================================
def view_data(df):
    """Menampilkan struktur dan isi data sebelum diolah."""
    print("Skema data (nama kolom dan tipenya):")
    df.printSchema()

    print("Contoh 5 baris (kolom teks panjang disembunyikan):")
    df.drop("cot_nutrition_logistics_rationale").show(5)

    print("Statistik ringkas kolom angka utama:")
    df.select(
        "delivery_distance_km",
        "transit_time_minutes",
        "food_arrival_temperature_celsius",
        "plate_waste_percentage",
    ).describe().show()

    print("Jumlah distribusi per klaster wilayah:")
    df.groupBy("regional_cluster").count().orderBy(F.desc("count")).show(truncate=False)


# =============================================================================
# 3. Operasi row (baris)
# =============================================================================
def row_operation(df):
    """Membersihkan baris: data kosong, duplikat, dan nilai tidak wajar."""
    jumlah_awal = df.count()

    # Buang baris yang punya nilai kosong (null)
    df = df.dropna()

    # Buang baris dengan ID distribusi yang sama (duplikat)
    df = df.dropDuplicates(["distribution_id"])

    # Simpan hanya baris dengan nilai yang masuk akal
    df = df.filter(
        (F.col("delivery_distance_km") > 0)                          # jarak harus positif
        & (F.col("transit_time_minutes") > 0)                        # waktu harus positif
        & (F.col("plate_waste_percentage").between(0, 100))         # persen: 0-100
        & (F.col("food_arrival_temperature_celsius").between(0, 100))
    )

    jumlah_akhir = df.count()
    print(f"Baris sebelum dibersihkan: {jumlah_awal:,}")
    print(f"Baris sesudah dibersihkan: {jumlah_akhir:,}")
    print(f"Baris yang dibuang       : {jumlah_awal - jumlah_akhir:,}")
    return df


# =============================================================================
# 4. Operasi column (kolom) + join
# =============================================================================
def column_operation(df, lookup):
    """Menambah kolom baru, membuang kolom yang tidak dipakai, dan join tabel pulau."""

    # Kolom baru 1: kategori waktu transit
    df = df.withColumn(
        "kategori_transit",
        F.when(F.col("transit_time_minutes") <= 15, "1. singkat (<=15 mnt)")
        .when(F.col("transit_time_minutes") <= 30, "2. sedang (16-30 mnt)")
        .otherwise("3. lama (>30 mnt)"),
    )

    # Kolom baru 2: apakah makanan tiba dalam suhu aman (standar BGN >= 60 °C)
    # 1 = aman, 0 = tidak aman
    df = df.withColumn(
        "suhu_aman",
        F.when(F.col("food_arrival_temperature_celsius") >= 60, 1).otherwise(0),
    )

    # Buang kolom yang tidak dipakai: teks panjang dan skor dari proses pembuatan data
    df = df.drop("cot_nutrition_logistics_rationale", "confidence_score")

    # Join: tambahkan kolom kelompok_pulau dan zona_waktu berdasarkan nama provinsi.
    # "left" = semua baris data utama tetap ada walaupun provinsinya tidak ditemukan.
    jumlah_sebelum = df.count()
    df = df.join(lookup, on="province", how="left")
    print(f"Baris sebelum join: {jumlah_sebelum:,} | sesudah join: {df.count():,}")

    df.select(
        "province", "kelompok_pulau", "transit_time_minutes",
        "kategori_transit", "food_arrival_temperature_celsius", "suhu_aman",
    ).show(5, truncate=False)
    return df


# =============================================================================
# 5. Fitur Spark: SQL
# =============================================================================
def sql_analysis(spark, df):
    """Menganalisis data dengan query SQL biasa."""

    # Daftarkan DataFrame sebagai tabel bernama "mbg" agar bisa dipakai di SQL
    df.createOrReplaceTempView("mbg")

    # Query 1: top 10 provinsi dengan rata-rata plate waste tertinggi.
    # Memakai rata-rata (AVG), bukan jumlah, karena banyaknya data per provinsi
    # tidak sama (provinsi di Jawa ±1.500 baris, provinsi kecil ±115 baris).
    top10_waste = spark.sql("""
        SELECT province,
               COUNT(*)                              AS jumlah,
               ROUND(AVG(plate_waste_percentage), 2) AS rata2_waste_persen
        FROM mbg
        GROUP BY province
        ORDER BY rata2_waste_persen DESC
        LIMIT 10
    """)
    print("Query 1 - Top 10 provinsi dengan plate waste tertinggi:")
    top10_waste.show(truncate=False)

    # Query 2: ranking provinsi di dalam setiap klaster (window function).
    # - Bagian WITH ... AS membuat tabel sementara "per_provinsi" berisi
    #   rata-rata waste per klaster + provinsi.
    # - RANK() OVER (PARTITION BY regional_cluster ORDER BY ... DESC) memberi
    #   peringkat 1, 2, 3, ... yang dihitung ULANG untuk setiap klaster.
    # - Berbeda dengan GROUP BY, window function tidak meringkas baris:
    #   setiap provinsi tetap satu baris, hanya ditambah kolom peringkat.
    # - Hanya 3 teratas per klaster yang ditampilkan agar hasil tidak terlalu panjang.
    ranking_klaster = spark.sql("""
        WITH per_provinsi AS (
            SELECT regional_cluster,
                   province,
                   COUNT(*)                              AS jumlah,
                   ROUND(AVG(plate_waste_percentage), 2) AS rata2_waste_persen
            FROM mbg
            GROUP BY regional_cluster, province
        ),
        diberi_peringkat AS (
            SELECT *,
                   RANK() OVER (PARTITION BY regional_cluster
                                ORDER BY rata2_waste_persen DESC) AS peringkat
            FROM per_provinsi
        )
        SELECT *
        FROM diberi_peringkat
        WHERE peringkat <= 3
        ORDER BY regional_cluster, peringkat
    """)
    print("Query 2 - Ranking provinsi dalam klaster (3 teratas per klaster):")
    ranking_klaster.show(truncate=False)

    # Query 3: perbandingan biaya antar tier kepatuhan.
    # Apakah distribusi yang tidak patuh standar memakai biaya lebih besar/kecil?
    biaya_tier = spark.sql("""
        SELECT bgn_program_compliance_tier                    AS tier,
               COUNT(*)                                       AS jumlah,
               ROUND(AVG(raw_ingredient_cost_per_portion_idr), 0) AS rata2_biaya_bahan,
               ROUND(AVG(total_meal_cost_idr), 0)             AS rata2_total_biaya,
               ROUND(AVG(plate_waste_percentage), 2)          AS rata2_waste_persen
        FROM mbg
        GROUP BY bgn_program_compliance_tier
        ORDER BY rata2_waste_persen
    """)
    print("Query 3 - Perbandingan biaya antar tier kepatuhan:")
    biaya_tier.show(truncate=False)

    # Query 4: persentase distribusi non-compliant per kelompok pulau.
    # CASE WHEN memberi nilai 1 untuk baris non-compliant dan 0 untuk lainnya,
    # lalu SUM / COUNT * 100 = persentase.
    # Ditambah rata-rata transit dan % suhu aman untuk melihat kaitannya dengan logistik.
    noncompliant_pulau = spark.sql("""
        SELECT kelompok_pulau,
               COUNT(*) AS jumlah,
               ROUND(SUM(CASE WHEN bgn_program_compliance_tier
                                   = 'Non-Compliant-Intervention-Required'
                              THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                   AS persen_non_compliant,
               ROUND(AVG(transit_time_minutes), 1) AS rata2_transit_menit,
               ROUND(AVG(suhu_aman) * 100, 1)      AS persen_suhu_aman
        FROM mbg
        GROUP BY kelompok_pulau
        ORDER BY persen_non_compliant DESC
    """)
    print("Query 4 - Persentase non-compliant per kelompok pulau:")
    noncompliant_pulau.show(truncate=False)


# =============================================================================
# 6. Fitur Spark: MLlib (machine learning)
# =============================================================================
# Fitur (input) model. Kolom berikut SENGAJA tidak dipakai karena dihitung dari
# plate waste atau dari kolom lain, sehingga model akan "menyontek" jawaban:
# waste_adjusted_effective_cost_idr, cost_to_nutrition_ratio,
# transit_cooling_rate_c_per_min, spoilage_risk_index, primary_food_waste_driver.
FITUR_KATEGORI = ["regional_cluster", "delivery_transport_mode", "protein_type"]
FITUR_ANGKA = [
    "delivery_distance_km",
    "transit_time_minutes",
    "food_arrival_temperature_celsius",
    "total_meal_cost_idr",
    "estimated_calories_kcal",
    "estimated_protein_grams",
]
TARGET = "plate_waste_percentage"


def ml_model(df):
    """Melatih Random Forest untuk memprediksi plate waste dan melihat faktor terpenting."""

    # Bagi data: 80% untuk melatih model, 20% untuk menguji.
    # seed=42 membuat pembagian selalu sama setiap program dijalankan.
    data_latih, data_uji = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Data latih: {data_latih.count():,} baris | data uji: {data_uji.count():,} baris")

    # Tahap 1: ubah kolom teks (kategori) menjadi angka, misal "Motor" -> 0, "Van" -> 1
    kolom_index = [k + "_index" for k in FITUR_KATEGORI]
    indexer = StringIndexer(inputCols=FITUR_KATEGORI, outputCols=kolom_index)

    # Tahap 2: gabungkan semua fitur menjadi satu kolom vektor bernama "features"
    # (format input yang diminta MLlib)
    semua_fitur = kolom_index + FITUR_ANGKA
    assembler = VectorAssembler(inputCols=semua_fitur, outputCol="features")

    # Tahap 3: model Random Forest = 100 pohon keputusan, hasilnya dirata-rata
    rf = RandomForestRegressor(
        featuresCol="features", labelCol=TARGET, numTrees=100, maxDepth=8, seed=42,
    )

    # Pipeline menjalankan ketiga tahap secara berurutan
    pipeline = Pipeline(stages=[indexer, assembler, rf])
    model = pipeline.fit(data_latih)          # latih model
    prediksi = model.transform(data_uji)      # prediksi data uji

    # Evaluasi: seberapa dekat prediksi dengan nilai sebenarnya
    evaluator = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction")
    rmse = evaluator.evaluate(prediksi, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(prediksi, {evaluator.metricName: "r2"})
    print(f"RMSE: {rmse:.2f} poin persen (rata-rata besar kesalahan prediksi)")
    print(f"R²  : {r2:.3f} (porsi variasi plate waste yang bisa dijelaskan model, 0-1)")

    print("\nContoh prediksi vs nilai sebenarnya:")
    prediksi.select("province", TARGET, F.round("prediction", 2).alias("prediksi")).show(5)

    # Feature importance: seberapa besar peran tiap fitur dalam prediksi (total = 1)
    nilai_importance = model.stages[-1].featureImportances.toArray()
    importance = sorted(zip(FITUR_KATEGORI + FITUR_ANGKA, nilai_importance),
                        key=lambda x: x[1], reverse=True)
    print("Faktor terpenting untuk plate waste:")
    for nama, nilai in importance:
        print(f"  {nama:<35} {nilai:.3f}")
    return importance


# =============================================================================
# 7. Solusi: grafik untuk laporan
# =============================================================================
def simpan_grafik(importance):
    """Membuat grafik feature importance dan menyimpannya ke folder outputs/."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Urutan dibalik agar faktor terbesar tampil paling atas
    nama = [x[0] for x in importance][::-1]
    nilai = [x[1] for x in importance][::-1]

    plt.figure(figsize=(7, 4))
    plt.barh(nama, nilai, color="#2a78d6")
    plt.title("Faktor terpenting untuk plate waste (Random Forest)")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "feature_importance.png", dpi=150)
    plt.close()

    print(f"Grafik disimpan: {OUTPUT_FOLDER.name}/feature_importance.png")


# =============================================================================
# Program utama: menjalankan semua langkah secara berurutan
# =============================================================================
def main():
    spark = init_spark()

    title("1. DATASET")
    df, lookup = read_data(spark)

    title("2. TAMPILAN DATAFRAME")
    view_data(df)

    title("3. OPERASI ROW")
    df = row_operation(df)

    title("4. OPERASI COLUMN + JOIN")
    df = column_operation(df, lookup)
    df.cache()  # simpan di memori karena dipakai berkali-kali di langkah berikutnya

    title("5. SPARK SQL")
    sql_analysis(spark, df)

    title("6. MLLIB - RANDOM FOREST")
    importance = ml_model(df)

    title("7. GRAFIK FEATURE IMPORTANCE")
    simpan_grafik(importance)

    spark.stop()


if __name__ == "__main__":
    main()
