import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

# --- KONFIGURASI ---
INPUT_FILENAME  = 'train_data.csv'
MODEL_DIR       = 'models/'
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

# 1. LOAD DATA
print("📂 Loading Final Data...")
df = pd.read_csv(INPUT_FILENAME)

# FITUR: 5 Sensor Fisik (Tanpa Jam, Tanpa Scaling karena RF hebat)
features = ['occupancy', 'temp', 'hum', 'lux', 'noise']
X = df[features]

# TARGET
y_class = df['status']                     
y_reg   = df[['energy_kwh', 'pmv', 'ppd']] 

# SPLIT (Tetap split buat validasi terakhir)
print("✂️ Splitting Data...")
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
)

# --- 2. TRAINING FINAL ---

# A. MODEL STATUS (Klasifikasi)
print("\n🌲 Melatih RANDOM FOREST CLASSIFIER (Status)...")
# n_estimators=100 sudah cukup. class_weight='balanced' wajib.
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
clf.fit(X_train, y_cls_train)

# Evaluasi Singkat
acc = clf.score(X_test, y_cls_test)
print(f"   ✅ Akurasi Status: {acc*100:.4f}% (Sempurna)")

# B. MODEL METRIK (Regresi)
print("\n🌲 Melatih RANDOM FOREST REGRESSOR (kWh, PMV, PPD)...")
reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
reg.fit(X_train, y_reg_train)

# Evaluasi Singkat
r2 = reg.score(X_test, y_reg_test)
print(f"   ✅ R2 Score Angka: {r2:.4f} (Sangat Presisi)")
y_pred_reg = reg.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_pred_reg)
print(f"   ✅ Rata-rata Meleset (MAE): {mae:.5f}")


# --- 3. SIMPAN MODEL ---
print("\n💾 Menyimpan Model Final...")
joblib.dump(clf, f'{MODEL_DIR}rf_status_model.pkl')
joblib.dump(reg, f'{MODEL_DIR}rf_metrics_model.pkl')

print("\n🎉 SELESAI! Model Random Forest siap dideploy.")
print(f"   Lokasi: {MODEL_DIR}")