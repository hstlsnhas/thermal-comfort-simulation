import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONFIGURASI ---
FILENAME = 'train_data.csv'

print(f"📂 Membaca file '{FILENAME}'...")
if not os.path.exists(FILENAME):
    print("❌ File tidak ditemukan! Generate dulu datanya.")
    exit()

df = pd.read_csv(FILENAME)

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

ax = sns.countplot(
    x='status', 
    data=df, 
    order=df['status'].value_counts().index, 
    palette='viridis',
    hue='status', 
    legend=False
)

# Judul dan Label
plt.title('Distribusi Status Ruangan (Class Balance)', fontsize=14, fontweight='bold')
plt.xlabel('Kategori Status', fontsize=12)
plt.ylabel('Jumlah Data', fontsize=12)
plt.xticks(rotation=45)

# --- MENAMBAHKAN LABEL ANGKA & PERSENTASE ---
total = len(df)
for container in ax.containers:
    labels = [f'{v.get_height()} ({v.get_height()/total*100:.1f}%)' for v in container]
    ax.bar_label(container, labels=labels, padding=3, fontsize=10)

plt.tight_layout()
output_img = 'visualisasi_status_distribusi.png'
plt.savefig(output_img, dpi=300)
print(f"✅ Grafik berhasil disimpan ke '{output_img}'")
plt.show()