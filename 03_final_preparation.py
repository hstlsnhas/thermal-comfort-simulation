import pandas as pd
import numpy as np
import os

# --- KONFIGURASI ---
INPUT_FILENAME  = 'clean_data.csv'
OUTPUT_FILENAME = 'train_data.csv'
GAP_START       = "2025-12-24 00:00:00"
GAP_END         = "2025-12-28 23:59:59"

# --- FUNGSI KALKULASI ---
def get_kwh_estimation(t):
    if pd.isna(t): return 0.0
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
    occ = row['occupancy']
    t   = row['temp']
    h   = row['hum']
    l   = row['lux']
    n   = row['noise']
    
    # --- 1. CEK KONDISI RUANGAN KOSONG (OCC = 0) ---
    if (occ == 0):
        # BOROS: Jika kosong tapi Suhu dingin (AC Nyala <= 23.5) ATAU Lampu Terang (> 150)
        if (t <= 23.5) and (l > 150): 
            return "Boros Energi", 0.0, 5
        # IDEAL: Jika kosong, Suhu Panas (AC Mati) DAN Gelap (Lampu Mati)
        elif (t > 23.5) and (l <= 150):
            return "Ideal", 0.0, 5 
        # Sisa (Misal kosong tapi gelap & dingin, atau panas & terang) -> Buang
        else: 
            return "Invalid", 0.0, 0

    # --- 2. CEK KASUS ADA ORANG TAPI GELAP (LUX RENDAH) ---
    if (occ > 0) and (l < 290):
            if (25.8 <= t <= 27.1): 
                pmv = 0.0; ppd = 5   # Gelap dan hangat nyaman
            elif (22.7 <= t < 25.8): 
                pmv = 0.0; ppd = 5   # Gelap dan nyaman optimal
            elif (19.5 <= t < 22.7): 
                pmv = -1.0; ppd = 25 # Gelap tapi Sejuk nyaman
            elif (t < 19.5):        
                pmv = -2.0; ppd = 75 # Gelap dan Kedinginan
            else:                   
                pmv = 2.0; ppd = 75  # Gelap dan Kepanasan
            
            return "Peringatan", pmv, ppd

    # --- 3. KATEGORI IDEAL (1-18 Orang) ---
    if (1 <= occ <= 18):
        if (19.5 <= t <= 25.5) and (40 <= h <= 65) and (290 <= l <= 500) and (n < 55):
            return "Ideal", 0.0, 5

    # --- 4. KATEGORI OPTIMALISASI (19-25 Orang) ---
    if (19 <= occ <= 25):
        if (24.0 <= t <= 27.0) and (290 <= l <= 450): # Hum & Noise dilonggarkan dikit
            return "Optimalisasi", 0.5, 10

    # --- 5. KATEGORI PERINGATAN (26-60 Orang) ---
    # Peringatan Level 1 (26-30 Orang)
    if (26 <= occ <= 30) and (26.3 < t <= 27.3) and (60 < h <= 75) and (430 < l <= 560):
        return "Peringatan", 1.0, 25
        
    # Peringatan Level 2 (31-60 Orang)
    if (31 <= occ <= 60) and (27.0 < t <= 28.8) and (65 < h <= 80) and (540 < l <= 660):
        return "Peringatan", 1.5, 50

    # --- 6. KATEGORI KRITIS (> 60 Orang) ---
    if (occ > 60) and (t > 28.3) and (h > 70) and (l > 640):
        return "Kritis", 2.5, 90
    
    # --- SISA (INVALID) ---
    return "Invalid", 0.0, 0

def generate_occupancy_from_real_sensors(row):
    """
    Mengecek suhu/sensor asli, lalu memberikan jumlah orang yang COCOK.
    """
    t = row['temp']
    h = row['hum']
    
    # Prioritas 1: Cek kondisi Panas (Untuk Kritis/Peringatan)
    if (t > 28.3): 
        # Suhu sangat panas -> Pasti rame banget
        return np.random.randint(61, 90) # Kritis
    
    if (27.0 < t <= 28.3):
        # Suhu agak panas -> Rame
        return np.random.randint(31, 60) # Peringatan Atas
        
    if (26.3 < t <= 27.0):
        # Suhu hangat -> Lumayan rame
        return np.random.randint(26, 30) # Peringatan Bawah

    # Prioritas 2: Cek kondisi Nyaman (Optimal/Ideal)
    if (24.8 < t <= 26.3):
        return np.random.randint(19, 25) # Optimalisasi
        
    if (20.5 < t <= 24.8):
        return np.random.randint(1, 18)  # Ideal
        
    # Prioritas 3: Cek kondisi Dingin (Boros)
    if (t <= 20.5):
        return 0 # Boros Energi
    
    # Fallback (Jika ada range suhu aneh yang terlewat, anggap Ideal)
    return np.random.randint(1, 18)


# --- MAIN PROGRAM ---
print("🚀 Memulai Proses...")

# 1. Load Data Asli
if not os.path.exists(INPUT_FILENAME):
    print("❌ File tidak ditemukan."); exit()
df_orig = pd.read_csv(INPUT_FILENAME)
print(f"📂 Data Asli dimuat: {len(df_orig):,} baris.")

if 'timestamp' in df_orig.columns:
    df_orig['timestamp'] = pd.to_datetime(df_orig['timestamp'])

# Pastikan kolom sensor ada
for col in ['temp', 'hum', 'lux', 'noise']:
    if col not in df_orig.columns:
        print(f"⚠️ Kolom {col} tidak ada, generate random wajar...")
        df_orig[col] = np.random.uniform(20, 30, len(df_orig)) # Random range luas

print("🧠 Menganalisa Sensor Asli (Temp) untuk menentukan Occupancy...")
df_orig['occupancy'] = df_orig.apply(generate_occupancy_from_real_sensors, axis=1)


# 2. GENERATE GAP DATA (PENYEIMBANG KASUS)
print("🧩 Generating Gap Data...")

timestamps_key = pd.date_range(start=GAP_START, end=GAP_END, freq="5min")

scenarios = {
    # Format: [OccMin, OccMax, T_Min, T_Max, H_Min, H_Max, L_Min, L_Max, N_Min, N_Max]
    'Kritis':      [61, 90,   28.5, 31.0,  72, 85,  650, 750,  58, 75], # Panas & Rame
    'Peringatan':  [31, 60,   27.2, 28.5,  66, 78,  545, 650,  45, 60], # Agak Panas
    'Optimal':     [19, 25,   25.0, 26.5,  52, 68,  300, 390,  42, 58], # Nyaman
    'Ideal':       [1,  18,   21.0, 24.0,  45, 55,  390, 440,  35, 45], # Sangat Nyaman
    'Boros':       [0,  0,    18.0, 20.0,  42, 58,  440, 500,  30, 48]  # Dingin tapi KOSONG (0 orang)
}

data_keys = []
for ts in timestamps_key:
    key = np.random.choice(
        ['Kritis', 'Peringatan', 'Optimal', 'Ideal', 'Boros'], 
        p=[0.25, 0.25, 0.20, 0.10, 0.20] 
    )
    
    p = scenarios[key]
    data_keys.append([
        ts, 
        np.random.randint(p[0], p[1]+1), # Occupancy
        np.random.uniform(p[2], p[3]),   # Temp
        np.random.uniform(p[4], p[5]),   # Hum
        np.random.uniform(p[6], p[7]),   # Lux
        np.random.uniform(p[8], p[9])    # Noise
    ])

df_gap = pd.DataFrame(data_keys, columns=['timestamp', 'occupancy', 'temp', 'hum', 'lux', 'noise'])
df_gap.set_index('timestamp', inplace=True)
df_gap = df_gap.resample('s').interpolate(method='linear').reset_index()
df_gap['occupancy'] = df_gap['occupancy'].round().astype(int)


# 3. MERGE
print("🔗 Menggabungkan Data...")
df_final = pd.concat([df_orig, df_gap], ignore_index=True)
df_final['timestamp'] = pd.to_datetime(df_final['timestamp'], errors='coerce')
df_final.sort_values(by='timestamp', inplace=True)
df_final.dropna(subset=['timestamp', 'temp'], inplace=True)


# 4. CALCULATE
print("⚡ Menghitung Status...")
df_final['energy_kwh'] = df_final['temp'].apply(get_kwh_estimation)
status_results = [get_compliance_status(row) for idx, row in df_final.iterrows()]
df_final['status'], df_final['pmv'], df_final['ppd'] = zip(*status_results)

# Filter Invalid
jumlah_invalid = len(df_final[df_final['status'] == 'Invalid'])
if jumlah_invalid > 0:
    print(f"\n⚠️ Ditemukan {jumlah_invalid:,} baris dengan status 'Invalid'.")
    print("🗑️ Sedang menghapus data Invalid...")
    # Proses penghapusan
    df_final = df_final[df_final['status'] != 'Invalid']
    print(f"✅ Data Invalid berhasil dibuang. Sisa data bersih: {len(df_final):,} baris.")
else:
    print("\n✅ Selesai! Tidak ditemukan data 'Invalid' (Data sudah bersih 100%).")

# Rounding
cols = ['temp', 'hum', 'lux', 'noise', 'energy_kwh']
df_final[cols] = df_final[cols].round(2)

print("\n📊 Statistik Data Akhir:")

print("--- Statistik Numerik ---")
print(df_final.describe().round(2)) 

print("\n--- Distribusi Kelas (Jumlah & Persentase) ---")
stat = pd.concat([
    df_final['status'].value_counts().rename('Jumlah'),
    (df_final['status'].value_counts(normalize=True) * 100).round(2).astype(str) + '%'
], axis=1)
print(stat)

#print(f"\n💾 Menyimpan ke '{OUTPUT_FILENAME}'...")
#df_final.to_csv(OUTPUT_FILENAME, index=False)
#print("✅ SELESAI!")