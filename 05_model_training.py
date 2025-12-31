import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# --- IMPORT 4 MODEL ---
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

warnings.filterwarnings('ignore')

# --- KONFIGURASI ---
INPUT_FILENAME = 'train_data.csv' 

# --- 1. PERSIAPAN DATA ---
print("📂 Loading Data...")
if 'timestamp' in pd.read_csv(INPUT_FILENAME).columns:
    df = pd.read_csv(INPUT_FILENAME)
else:
    print("File tidak ditemukan atau format salah"); exit()

# FITUR
features = ['occupancy', 'temp', 'hum', 'lux', 'noise', 'luas']
X = df[features]

# TARGET
y_status = df['status']
y_reg    = df[['energy_kwh', 'pmv', 'ppd']]

# Encode Status ke Angka
le = LabelEncoder()
y_status_enc = le.fit_transform(y_status)
print(f"   Label Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split Data (Stratified)
print("✂️ Splitting Data (80/20)...")
X_train_raw, X_test_raw, y_stat_train, y_stat_test, y_reg_train, y_reg_test = train_test_split(
    X, y_status_enc, y_reg, test_size=0.2, random_state=42, stratify=y_status_enc
)

# --- 2. SCALING ---
print("⚖️ Melakukan Standardisasi Data...")
scaler = StandardScaler()

# Fit pada Training, Transform pada Training & Test
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)


# --- 3. DEFINISI 4 MODEL ---
models = [
    {
        'name': 'Decision Tree',
        'clf': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'reg': DecisionTreeRegressor(random_state=42)
    },
    {
        'name': 'Random Forest',
        'clf': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
        'reg': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    },
    {
        'name': 'XGBoost',
        'clf': XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1),
        'reg': MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1))
    },
    {
        'name': 'K-Nearest Neighbors (KNN)',
        'clf': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'reg': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    }
]

# --- 4. TRAINING ---
results = []

print("\n🥊 MULAI BENCHMARKING 4 MODEL...")
print("="*90)

for m in models:
    print(f"⚙️  Sedang melatih {m['name']}...", end=" ")
    start_time = time.time()
    
    # A. Latih Classifier (Status)
    m['clf'].fit(X_train, y_stat_train)
    pred_stat = m['clf'].predict(X_test)
    
    # B. Latih Regressor (Angka)
    m['reg'].fit(X_train, y_reg_train)
    pred_reg = m['reg'].predict(X_test)
    
    duration = time.time() - start_time
    print(f"✅ Selesai ({duration:.2f} detik)")
    
    # C. Hitung Skor
    # Klasifikasi
    acc = accuracy_score(y_stat_test, pred_stat)
    f1  = f1_score(y_stat_test, pred_stat, average='macro') 
    
    # Regresi (MAE, MSE, RMSE)
    mae_avg = mean_absolute_error(y_reg_test, pred_reg)
    mse_avg = mean_squared_error(y_reg_test, pred_reg)
    rmse_avg = np.sqrt(mse_avg)                
    r2_avg  = r2_score(y_reg_test, pred_reg)
    
    results.append({
        'Model': m['name'],
        'F1-Score (Status)': f1,        
        'Accuracy (Status)': acc,      
        'MAE (Angka)': mae_avg,
        'RMSE (Angka)': rmse_avg,
        'R2 Score (Angka)': r2_avg,     
        'Waktu (s)': duration
    })

# --- 5. TAMPILKAN HASIL ---
df_results = pd.DataFrame(results)

# Urutkan: F1 (Tertinggi) -> RMSE (Terendah)
# Kita pakai RMSE sebagai penentu kedua karena lebih sensitif terhadap outlier dibanding MAE
df_results = df_results.sort_values(by=['F1-Score (Status)', 'RMSE (Angka)'], ascending=[False, True])

print("\n📊 KLASEMEN AKHIR (Diurutkan F1 Tertinggi & RMSE Terendah):")
print(df_results.to_string(index=False))

# --- 6. VISUALISASI ---
print("\n🎨 Sedang membuat plot perbandingan...")

sns.set_style("whitegrid")
plt.figure(figsize=(16, 10)) # Ukuran kanvas besar

# --- PLOT 1: F1-Score (Klasifikasi) ---
plt.subplot(2, 2, 1) # Baris 1, Kolom 1
ax1 = sns.barplot(x='Model', y='F1-Score (Status)', data=df_results, palette='viridis')
plt.title('Perbandingan F1-Score (Status) - Semakin Tinggi Semakin Bagus', fontsize=12, fontweight='bold')
plt.ylim(0.9, 1.01) # Zoom in skala karena nilainya pasti tinggi semua (dekat 1.0)
plt.ylabel('F1 Score')

# Tambah label angka di atas batang
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 9), textcoords='offset points')

# --- PLOT 2: RMSE (Regresi) ---
plt.subplot(2, 2, 2) # Baris 1, Kolom 2
ax2 = sns.barplot(x='Model', y='RMSE (Angka)', data=df_results, palette='magma')
plt.title('Perbandingan RMSE (Error Angka) - Semakin Rendah Semakin Bagus', fontsize=12, fontweight='bold')
plt.ylabel('RMSE')

# Label angka
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 9), textcoords='offset points')

# --- PLOT 3: R2 Score (Akurasi Regresi) ---
plt.subplot(2, 2, 3) # Baris 2, Kolom 1
ax3 = sns.barplot(x='Model', y='R2 Score (Angka)', data=df_results, palette='rocket')
plt.title('Perbandingan R2 Score - Semakin Dekat 1.0 Semakin Bagus', fontsize=12, fontweight='bold')
plt.ylim(0.9, 1.01) # Zoom in
plt.ylabel('R2 Score')

# Label angka
for p in ax3.patches:
    ax3.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 9), textcoords='offset points')

# --- PLOT 4: MAE (Mean Absolute Error) ---
plt.subplot(2, 2, 4) # Baris 2, Kolom 2
ax4 = sns.barplot(x='Model', y='MAE (Angka)', data=df_results, palette='coolwarm')
plt.title('Mean Absolute Error (MAE) - Semakin Rendah Semakin Bagus', fontsize=12, fontweight='bold')
plt.ylabel('MAE')

# Label angka
for p in ax4.patches:
    ax4.annotate(f'{p.get_height():.5f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 9), textcoords='offset points')

plt.tight_layout()
output_img = 'hasil_perbandingan_model.png'
plt.savefig(output_img, dpi=300)
print(f"✅ Gambar grafik tersimpan sebagai '{output_img}'")
#plt.show()

# --- ANALISIS ---
best_model = df_results.iloc[0]
print("\n🏆 MODEL TERBAIK:")
print(f"Model: {best_model['Model']}")
print("-" * 30)