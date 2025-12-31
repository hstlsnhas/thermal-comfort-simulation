import pandas as pd
import numpy as np
import joblib
import os

# --- KONFIGURASI ---
MODEL_DIR = 'models/'
STATUS_MODEL = f'{MODEL_DIR}rf_status_model.pkl'
METRICS_MODEL = f'{MODEL_DIR}rf_metrics_model.pkl'

# Cek Model
if not os.path.exists(STATUS_MODEL) or not os.path.exists(METRICS_MODEL):
    print("❌ Model tidak ditemukan! Pastikan sudah run training dulu."); exit()

# 1. LOAD MODEL
print("📂 Memuat otak AI (Loading Models)...")
clf = joblib.load(STATUS_MODEL) # Model Klasifikasi
reg = joblib.load(METRICS_MODEL) # Model Regresi

# 2. BUAT DATA TEST (DUMMY)
print("🎲 Membuat Data Dummy untuk Testing...\n")

data_test = []

# A. SKENARIO KHUSUS
data_test.append({'Note': 'Tes Boros (Kosong, Dingin, Terang)', 'occupancy': 0,  'temp': 18.0, 'hum': 50, 'lux': 400, 'noise': 35})
data_test.append({'Note': 'Tes Ideal (Sedikit Orang, Nyaman)',  'occupancy': 5,  'temp': 24.0, 'hum': 55, 'lux': 350, 'noise': 45})
data_test.append({'Note': 'Tes Peringatan (Gelap tapi ada org)', 'occupancy': 10, 'temp': 24.0, 'hum': 60, 'lux': 100, 'noise': 40})
data_test.append({'Note': 'Tes Kritis (Rame & Panas)',          'occupancy': 80, 'temp': 30.0, 'hum': 85, 'lux': 700, 'noise': 75})

# B. SKENARIO RANDOM (5 Data Acak)
for i in range(5):
    data_test.append({
        'Note': f'Random Test #{i+1}',
        'occupancy': np.random.randint(0, 60),      # 0 s/d 60 orang
        'temp':      np.random.uniform(18.0, 32.0), # 18 s/d 32 Derajat
        'hum':       np.random.randint(40, 90),     # 40% s/d 90%
        'lux':       np.random.randint(0, 800),     # 0 s/d 800 Lux
        'noise':     np.random.randint(30, 80)      # 30 s/d 80 dB
    })

df_test = pd.DataFrame(data_test)
notes = df_test['Note']
X_test = df_test[['occupancy', 'temp', 'hum', 'lux', 'noise']]

# 3. LAKUKAN PREDIKSI
print("⚡ Sedang memprediksi...")
pred_status = clf.predict(X_test)
pred_metrics = reg.predict(X_test)

# 4. TAMPILKAN HASIL
print("="*100)
print(f"{'SKENARIO':<35} | {'INPUT (Occ, T, H, L, N)':<30} | {'STATUS PREDIKSI':<15} | {'METRIK (kWh, PMV, PPD)'}")
print("="*100)

for i in range(len(df_test)):
    # Ambil input
    row = X_test.iloc[i]
    input_str = f"{int(row['occupancy'])}, {row['temp']:.1f}, {int(row['hum'])}, {int(row['lux'])}, {int(row['noise'])}"
    
    # Ambil hasil prediksi
    stat = pred_status[i]
    kwh  = pred_metrics[i][0]
    pmv  = pred_metrics[i][1]
    ppd  = pred_metrics[i][2]
    
    # Formatting biar rapi
    metrics_str = f"{kwh:.2f} kWh, {pmv:.2f} PMV, {ppd:.1f}%"
    
    print(f"{notes[i]:<35} | {input_str:<30} | {stat:<15} | {metrics_str}")

print("="*100)