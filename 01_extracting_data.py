import pandas as pd
import json
import numpy as np

# --- KONFIGURASI ---
INPUT_FILENAME = 'raw_data.csv'
OUTPUT_FILENAME = 'extracted_data.csv'
LUAS_RUANGAN = 176.0

print("="*50)
print("🚀  EXTRACTING DATA IOT")
print("="*50)

# 1. LOAD DATA
print(f"\n📂  [1/2] Membaca file: {INPUT_FILENAME}...")
try:
    df = pd.read_csv(INPUT_FILENAME)
    print(f"    ✅ Berhasil! Ditemukan {len(df)} baris data mentah.")
except FileNotFoundError:
    print(f"    ❌ Error: File '{INPUT_FILENAME}' tidak ditemukan!")
    exit()

# 2. EKSTRAKSI FITUR
print("\n⚙️   [2/2] Mengekstrak data JSON (Payload)...")

def extract_features(row):
    try:
        # Menangani string JSON yang menggunakan tanda kutip tunggal
        payload_raw = row['payload']
        if isinstance(payload_raw, str):
            data = json.loads(payload_raw.replace("'", '"'))
        else:
            data = payload_raw
            
        sensor_type = row['sensor_name']
        result = {}
            
        if sensor_type == 'hvac':
            result['temp'] = float(data.get('temp', np.nan))
            result['hum'] = float(data.get('hum', np.nan))
            result['noise'] = float(data.get('noise', np.nan))
        elif sensor_type == 'lux-meter':
            result['lux'] = float(data.get('light_level', np.nan))
        return pd.Series(result)
    except Exception:
        return pd.Series({})

df_extracted = df.apply(extract_features, axis=1)

df_combined = pd.concat([df['timestamp'], df_extracted], axis=1)
df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
df_combined = df_combined.sort_values('timestamp')

df_combined.to_csv(OUTPUT_FILENAME, index=False)

print("\n" + "="*50)
print(f"✅  SELESAI! Total baris : {len(df_combined)}. Preview 5 baris teratas: ")
print("="*50)
print(df_combined.head(5))
print("\n")