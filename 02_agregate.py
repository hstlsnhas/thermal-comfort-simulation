import pandas as pd
import os

# --- KONFIGURASI ---
INPUT_FILENAME = 'extracted_data.csv'
OUTPUT_FILENAME = 'clean_data.csv'
THRESHOLD_RATIO = 0.3

print("="*50)
print("🧹   CLEANING & AGGREGATION")
print("="*50)

# 1. LOAD DATA
if not os.path.exists(INPUT_FILENAME):
    print(f"❌ Error: File '{INPUT_FILENAME}' tidak ditemukan!")
    exit()

df = pd.read_csv(INPUT_FILENAME)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. AGGREGATION
print("\n⚙️   [1/3] Melakukan Aggregasi per Detik (Tanpa mengisi gap)...")
df['timestamp'] = df['timestamp'].dt.floor('S')
df_clean = df.groupby('timestamp').mean()

# 3. MISSING VALUE
total_rows = len(df_clean)
rows_with_nan = df_clean.isnull().any(axis=1).sum()
missing_percentage = rows_with_nan / total_rows if total_rows > 0 else 0

print(f"    📊 Total Baris: {total_rows}")
print(f"    ⚠️ Baris dengan NaN: {rows_with_nan} ({missing_percentage:.1%})")

if missing_percentage < THRESHOLD_RATIO:
    print(f"\n Karena Missing Value < {THRESHOLD_RATIO*100}%, keputusan: DROP ROWS")
    df_final = df_clean.dropna()   
else:
    print(f"\n Karena Missing Value >= {THRESHOLD_RATIO*100}%, keputusan: FILL MEDIAN")
    median_vals = df_clean.median(numeric_only=True)
    df_final = df_clean.fillna(median_vals)

df_final = df_final.reset_index()

# 4. SAVE
print(f"\n💾  [3/3] Menyimpan hasil ke: {OUTPUT_FILENAME}...")
df_final.to_csv(OUTPUT_FILENAME, index=False)

print("\n" + "="*50)
print("✅  SELESAI!")
print("="*50)
print(df_final.head(10))
print("\n")