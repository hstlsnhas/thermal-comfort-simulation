import pandas as pd
import numpy as np
import os

# --- KONFIGURASI ---
INPUT_FILENAME  = 'clean_data.csv'
OUTPUT_FILENAME = 'train_data1.csv'

# --- FUNGSI KALKULASI ---
def get_kwh_estimation(t):
    """Menghitung estimasi konsumsi energi berdasarkan suhu."""
    if pd.isna(t): return 0.0 # Safety check
    if   18.0 <= t < 19.0: return 0.84
    elif 19.0 <= t < 20.0: return 0.80
    elif 20.0 <= t < 21.0: return 0.76
    elif 21.0 <= t < 22.0: return 0.71
    elif 22.0 <= t < 23.0: return 0.67
    elif 23.0 <= t < 24.0: return 0.63
    elif 24.0 <= t < 25.0: return 0.59
    elif 25.0 <= t < 26.0: return 0.50
    elif 26.0 <= t < 27.0: return 0.42
    elif 27.0 <= t < 28.0: return 0.34
    elif 28.0 <= t < 29.0: return 0.25
    elif 29.0 <= t < 30.0: return 0.17
    elif 30.0 <= t < 31.0: return 0.10
    elif 31.0 <= t < 32.0: return 0.05
    else: return 0.0

def get_compliance_status(row):
    # Tidak perlu handle NaN jadi 0 di sini karena data sudah di-drop sebelumnya
    occ = row['occupancy']
    t   = row['temp']
    h   = row['hum']
    l   = row['lux']
    n   = row['noise']
    
    # Logic Kepatuhan
    if (occ == 0):
        if (17.5 <= t <= 20.5) and (40 <= h <= 60) and (l >= 430) and (n < 50):
            return "Boros Energi", 0.0, 5
        else:
            return "Invalid", 0.0, 0
    if (1 <= occ <= 18) and (19.5 < t <= 25.2) and (40 <= h <= 60) and (380 <= l <= 450) and (n < 50):
        return "Ideal", 0.0, 5
    if (19 <= occ <= 25) and (24.8 < t <= 26.8) and (50 < h <= 70) and (290 <= l <= 400) and (40 <= n <= 60):
        return "Optimalisasi", 1.0, 25
    if (26 <= occ <= 30) and (26.3 < t <= 27.3) and (60 < h <= 75) and (430 < l <= 560) and (40 <= n <= 65):
        return "Peringatan", 1.0, 25
    if (31 <= occ <= 60) and (27.0 < t <= 28.8) and (65 < h <= 80) and (540 < l <= 660) and (40 <= n <= 65):
        return "Peringatan", 1.0, 75
    if (occ > 60) and (t > 28.3) and (h > 70) and (l > 640) and (n > 55):
        return "Kritis", 2.0, 90
    return "Invalid", 0.0, 0


# --- MAIN PROGRAM ---

print("🚀 Memulai proses data asli...")

# 1. Load Data Asli
if not os.path.exists(INPUT_FILENAME):
    print(f"❌ Error: File '{INPUT_FILENAME}' tidak ditemukan.")
    exit()

df_final = pd.read_csv(INPUT_FILENAME)
print(f"📂 Data Awal dimuat: {len(df_final):,} baris.")

# 2. Generate Kolom Tambahan (Occupancy, Hum, Lux, Noise) jika SEKUENS KOLOM hilang
# Catatan: Ini hanya generate jika KOLOM tidak ada. Jika kolom ada tapi isinya kosong (NaN), akan didrop nanti.
print("🎲 Mengecek kelengkapan kolom...")

if 'occupancy' not in df_final.columns:
    print("   -> Generating Kolom Occupancy (Randomized Rules)...")
    rules = [(0,10,0.1), (11,18,0.2), (19,25,0.2), (26,30,0.2), (31,40,0.15), (41,60,0.15)]
    min_vals=[r[0] for r in rules]; max_vals=[r[1] for r in rules]; probs=[r[2] for r in rules]
    cat_ids = np.random.choice(range(len(rules)), size=len(df_final), p=probs)
    final_occ = np.zeros(len(df_final), dtype=int)
    for i in range(len(rules)):
        mask = (cat_ids == i)
        if mask.sum() > 0:
            final_occ[mask] = np.random.randint(min_vals[i], max_vals[i]+1, size=mask.sum())
    df_final['occupancy'] = final_occ

if 'hum' not in df_final.columns:
    df_final['hum'] = np.random.randint(45, 65, size=len(df_final))
if 'lux' not in df_final.columns:
    df_final['lux'] = np.random.randint(300, 500, size=len(df_final))
if 'noise' not in df_final.columns:
    df_final['noise'] = np.random.randint(35, 55, size=len(df_final))


# 3. Cleaning & Formatting
print("🧹 Membersihkan data (Convert Numeric & Drop NaN)...")

if 'timestamp' in df_final.columns:
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'], errors='coerce')
    df_final.dropna(subset=['timestamp'], inplace=True)
    df_final.sort_values(by='timestamp', ascending=True, inplace=True)

if 'temp' in df_final.columns:
    # Coerce error jadi NaN agar bisa didrop
    df_final['temp'] = pd.to_numeric(df_final['temp'], errors='coerce')

# --- LOGIC DROP NAN (Revisi di sini) ---
# Menentukan kolom yang wajib ada isinya
cols_to_check = ['occupancy', 'temp', 'hum', 'lux', 'noise']
cols_exist = [c for c in cols_to_check if c in df_final.columns]

initial_count = len(df_final)
df_final.dropna(subset=cols_exist, inplace=True)
dropped_count = initial_count - len(df_final)

if dropped_count > 0:
    print(f"⚠️  Dihapus {dropped_count} baris yang mengandung nilai NaN/Null.")
else:
    print("✅ Data bersih (tidak ada NaN yang dihapus).")


# 4. Final Calculation
print("⚡ Menghitung Estimasi kWh & Status...")
df_final['energy_kwh'] = df_final['temp'].apply(get_kwh_estimation)

status_results = [get_compliance_status(row) for idx, row in df_final.iterrows()]
df_final['status'], df_final['pmv'], df_final['ppd'] = zip(*status_results)


# 5. Pembulatan
cols_to_round = ['temp', 'hum', 'lux', 'noise', 'energy_kwh']
cols_exist_round = [c for c in cols_to_round if c in df_final.columns]
df_final[cols_exist_round] = df_final[cols_exist_round].round(2)


# 6. Simpan Hasil
print("\n📊 Statistik Data Akhir:")
print(df_final[['status']].value_counts())
print("\n🔍 Preview Data:")
print(df_final.head())

print(f"\n💾 Menyimpan ke '{OUTPUT_FILENAME}'...")
df_final.to_csv(OUTPUT_FILENAME, index=False)
print("✅ SELESAI! Data siap.")