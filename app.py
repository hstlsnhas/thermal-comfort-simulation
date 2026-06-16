import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import math
import os
import joblib
import google.generativeai as genai
import time

# ==============================================================================
# KONFIGURASI PANDAS STYLER (Menghindari StreamlitAPIException pada Data Besar)
# ==============================================================================
pd.set_option("styler.render.max_elements", 4000000)

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="SensiRoom - AI Smart Room Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# INISIALISASI SESSION STATE UNTUK INPUT WIDGET & REKOMENDASI AI
# ==============================================================================
if "input_occupancy" not in st.session_state:
    st.session_state.input_occupancy = 2
if "input_luas" not in st.session_state:
    st.session_state.input_luas = 176
if "input_temp" not in st.session_state:
    st.session_state.input_temp = 24.0
if "input_hum" not in st.session_state:
    st.session_state.input_hum = 50.0
if "input_lux" not in st.session_state:
    st.session_state.input_lux = 350
if "input_noise" not in st.session_state:
    st.session_state.input_noise = 45.0
if "gemini_result" not in st.session_state:
    st.session_state.gemini_result = None
if "show_reset_success" not in st.session_state:
    st.session_state.show_reset_success = False

# ==============================================================================
# CALLBACK RESETER (Mencegah StreamlitAPIException)
# ==============================================================================
def reset_all_parameters():
    """
    Fungsi callback yang dipanggil sebelum komponen dirender ulang.
    Sangat aman digunakan untuk mengubah session_state yang terikat pada key widget.
    """
    st.session_state.input_occupancy = 2
    st.session_state.input_luas = 176
    st.session_state.input_temp = 24.0
    st.session_state.input_hum = 50.0
    st.session_state.input_lux = 350
    st.session_state.input_noise = 45.0
    st.session_state.gemini_result = None
    st.session_state.show_reset_success = True

# ==============================================================================
# CUSTOM CSS (Premium Dark Mode, Typography & Glassmorphism)
# ==============================================================================
st.markdown("""
<style>
/* ================== FONT & GLOBAL STYLE ================== */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #0b0a14;
    color: #f1f5f9;
}

/* Optimasi Header & Jarak Streamlit */
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* Hide Default Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}

/* ================== HEADER BRANDING ================== */
.branding-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    background: rgba(18, 17, 32, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.brand-logo {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7 0%, #3b82f6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.brand-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 500;
    text-align: right;
}

/* ================== STYLED BUTTON NAV ================== */
div.stButton > button {
    background-color: rgba(255, 255, 255, 0.03);
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.25s ease-in-out;
    width: 100%;
}
div.stButton > button:hover {
    border-color: #a855f7;
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(59, 130, 246, 0.1));
    color: #ffffff;
    box-shadow: 0 4px 20px rgba(168, 85, 247, 0.15);
}

/* ================== HERO ================== */
.hero-section {
    background: radial-gradient(circle at 90% 10%, rgba(168, 85, 247, 0.12) 0%, rgba(59, 130, 246, 0.04) 50%, transparent 100%);
    background-color: rgba(18, 17, 32, 0.45);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 2.8rem 2.5rem;
    margin-bottom: 2rem;
}
.gradient-text {
    background: linear-gradient(135deg, #c084fc 0%, #6366f1 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* ================== CARDS ================== */
.custom-card {
    background: rgba(18, 17, 32, 0.6);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 1.8rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    height: 100%;
}
.custom-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168, 85, 247, 0.3);
    box-shadow: 0 15px 30px rgba(168, 85, 247, 0.12);
}
.card-icon {
    font-size: 2.2rem;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #a855f7, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}
.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    color: #f8fafc;
}
.card-desc {
    font-size: 0.95rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ================== METRIC CARDS ================== */
.metric-card {
    background: linear-gradient(145deg, rgba(20, 19, 38, 0.8), rgba(12, 11, 24, 0.8));
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 1.6rem 1.2rem;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    position: relative;
    overflow: hidden;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #a855f7, #3b82f6, #ec4899);
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 0.6rem;
    letter-spacing: -0.5px;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* ================== FORM OVERRIDES ================== */
div[data-testid="stForm"] {
    background: rgba(15, 14, 28, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 22px !important;
    padding: 2.2rem !important;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3) !important;
}

/* ================== PMV SCALE BAR ================== */
.pmv-container {
    margin: 1.5rem 0;
    padding: 1.8rem;
    background: rgba(15, 14, 28, 0.7);
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.pmv-track {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(90deg, #3b82f6 0%, #10b981 50%, #ef4444 100%);
    position: relative;
    margin: 1.5rem 0;
}
.pmv-indicator {
    width: 24px;
    height: 24px;
    background-color: #ffffff;
    border: 4px solid #a855f7;
    border-radius: 50%;
    position: absolute;
    top: -6px;
    transform: translateX(-50%);
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.6);
}
.pmv-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 600;
    padding: 0 5px;
}

/* ================== INSIGHT LISTS ================== */
.insight-list {
    list-style-type: none;
    padding-left: 0;
    margin: 0;
}
.insight-item {
    position: relative;
    padding-left: 1.8rem;
    margin-bottom: 0.9rem;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #cbd5e1;
}
.insight-item::before {
    content: '✦';
    position: absolute;
    left: 0;
    color: #a855f7;
    font-weight: bold;
}

/* ================== BADGES ================== */
.source-badge {
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.25);
    color: #c084fc;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# INPUT API KEY GEMINI DI SIDEBAR
# ==============================================================================
st.sidebar.markdown("### ⚙️ Konfigurasi API")
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Masukkan API Key Gemini untuk mengaktifkan AI Insights dinamis berbasis LLM."
)
st.sidebar.markdown("---")

# ==============================================================================
# FUNGSI INTEGRASI GEMINI API (MENGGUNAKAN LIBRARY RESMI GOOGLE-GENERATIVEAI)
# ==============================================================================
def get_gemini_insights(prompt, system_instruction, api_key):
    """
    Menggunakan library resmi `google-generativeai` untuk memanggil model `gemini-2.5-flash`.
    Menerapkan penanganan kesalahan cerdas dan retry jika terjadi rate-limiting (429).
    """
    try:
        # Konfigurasi kunci API secara global dalam session thread ini
        genai.configure(api_key=api_key)
        
        # Inisialisasi Model dengan system instruction bawaan
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction
        )
        
        # Percobaan pemanggilan dengan backoff manual sederhana untuk rate limit
        delays = [1, 2, 4]
        for attempt, delay in enumerate(delays):
            try:
                response = model.generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as inner_e:
                # Jika terkena kuota batas laju (exhausted), tunggu sebentar
                if "429" in str(inner_e) or "exhausted" in str(inner_e).lower():
                    time.sleep(delay)
                    continue
                # Lempar kesalahan jika jenis kesalahan lainnya
                raise inner_e
                
        return "❌ Gagal memuat rekomendasi. Terjadi penolakan batas laju (Rate Limit) berulang pada API Key Anda."

    except Exception as e:
        error_msg = str(e)
        # Deteksi kesalahan API Key yang salah/invalid secara eksplisit
        if "API_KEY_INVALID" in error_msg or "400" in error_msg or "key" in error_msg.lower():
            return (
                "❌ **API Key Tidak Valid atau Salah!**\n\n"
                "Sistem mendeteksi bahwa API Key Gemini yang Anda masukkan di sidebar salah atau "
                "tidak dikenali oleh Google AI Studio. Silakan periksa kembali kunci Anda."
            )
            
        return (
            f"❌ **Gagal Memuat AI Insights dari Gemini API.**\n\n"
            f"Berikut adalah detail kesalahan dari library `google-generativeai`:\n"
            f"```text\n{error_msg}\n```\n"
            f"**Rekomendasi Langkah Penyembuhan:**\n"
            f"1. Pastikan Anda sudah menginstal library dengan menjalankan: `pip install google-generativeai`.\n"
            f"2. Pastikan komputer Anda terhubung ke internet dengan stabil."
        )

# ==============================================================================
# INTEGRASI DATA RIIL & SIMULASI PENYEIMBANG (Final Prep Replicated)
# ==============================================================================
@st.cache_data
def load_production_data():
    """
    Membaca dataset utama 'train_data.csv'. 
    Jika tidak ada di sistem, ia akan meng-generate dataset sintetis yang identik secara cerdas
    menggunakan formula persis dari script penyiapan data Anda.
    """
    input_filename = 'train_data.csv'
    
    if os.path.exists(input_filename):
        df = pd.read_csv(input_filename)
        return df, "Real Production Dataset (train_data.csv)"
    
    # === REPLIKASI GENERATOR DATA APABILA FILE TIDAK ADA ===
    np.random.seed(42)
    n_samples = 1500
    
    # Skema simulasi sesuai scenario final prep Anda
    scenarios = {
        'Kritis':      [61, 90,   28.5, 31.0,  72, 85,  650, 750,  58, 75], 
        'Peringatan':  [31, 60,   27.2, 28.5,  66, 78,  545, 650,  45, 60], 
        'Optimal':      [19, 25,   25.0, 26.5,  52, 68,  300, 390,  42, 58], 
        'Ideal':       [1,  18,   21.0, 24.0,  45, 55,  390, 440,  35, 45], 
        'Boros':       [0,  0,    18.0, 20.0,  42, 58,  440, 500,  30, 48]  
    }
    
    # Probabilitas distribusi kelas
    keys = np.random.choice(
        ['Kritis', 'Peringatan', 'Optimal', 'Ideal', 'Boros'], 
        size=n_samples, 
        p=[0.25, 0.25, 0.20, 0.10, 0.20]
    )
    
    data = []
    for k in keys:
        p = scenarios[k]
        occ = np.random.randint(p[0], p[1]+1)
        t = np.round(np.random.uniform(p[2], p[3]), 2)
        h = np.round(np.random.uniform(p[4], p[5]), 2)
        l = np.round(np.random.uniform(p[6], p[7]), 2)
        n = np.round(np.random.uniform(p[8], p[9]), 2)
        
        # Kalkulasi estimasi kWh persis dari fungsi get_kwh_estimation Anda
        if 18.0 <= t < 19.0: kwh = 0.84
        elif 19.0 <= t < 20.0: kwh = 0.80
        elif 20.0 <= t < 21.0: kwh = 0.76
        elif 21.0 <= t < 22.0: kwh = 0.71
        elif 22.0 <= t < 23.0: kwh = 0.67
        elif 23.0 <= t < 24.0: kwh = 0.63
        elif 24.0 <= t < 25.0: kwh = 0.59
        elif 25.0 <= t < 26.0: kwh = 0.50
        elif 26.0 <= t < 27.0: kwh = 0.42
        elif 27.0 <= t < 28.0: kwh = 0.34
        elif 28.0 <= t < 29.0: kwh = 0.25
        elif 29.0 <= t < 30.0: kwh = 0.17
        elif 30.0 <= t < 31.0: kwh = 0.10
        elif 31.0 <= t < 32.0: kwh = 0.05
        else: kwh = 0.0
        
        # Hitung PMV & PPD secara simulasi
        if k == 'Kritis': pmv, ppd = 2.5, 90.0
        elif k == 'Peringatan': pmv, ppd = 1.25, 37.5
        elif k == 'Optimal': pmv, ppd = 0.5, 10.0
        elif k == 'Ideal': pmv, ppd = 0.0, 5.0
        else: pmv, ppd = 0.0, 5.0 # Boros
        
        data.append([occ, t, h, l, n, 176, kwh, k, pmv, ppd])
        
    df = pd.DataFrame(data, columns=['occupancy', 'temp', 'hum', 'lux', 'noise', 'luas', 'energy_kwh', 'status', 'pmv', 'ppd'])
    return df, "Simulated Real Dataset (Fall-back Mode)"

# Load data aktif
df_active, data_source_label = load_production_data()

# ==============================================================================
# SYSTEM LOADING / TRAINING MODEL MACHINE LEARNING
# ==============================================================================
@st.cache_resource
def load_or_train_models(df):
    """
    Memuat model klasifikasi dan regresi biner (.pkl).
    Jika model fisik tidak ditemukan di disk, maka akan langsung dilatih di RAM
    menggunakan parameter tepat dari script Anda (Random Forest).
    """
    model_dir = 'models/'
    clf_path = os.path.join(model_dir, 'rf_status_model.pkl')
    reg_path = os.path.join(model_dir, 'rf_energy_model.pkl')
    
    loaded_from_disk = False
    
    # 1. Coba load dari file terlebih dahulu
    if os.path.exists(clf_path) and os.path.exists(reg_path):
        try:
            clf = joblib.load(clf_path)
            reg = joblib.load(reg_path)
            loaded_from_disk = True
            return clf, reg, "Pre-trained RF Models loaded successfully from /models directory."
        except Exception as e:
            pass # Fall back to local train jika loading gagal
            
    # 2. Train model instan di RAM apabila file pkl absen
    X = df[['occupancy', 'temp', 'hum', 'lux', 'noise', 'luas']]
    y_class = df['status']
    y_reg = df[['energy_kwh', 'pmv', 'ppd']]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    clf.fit(X, y_class)
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X, y_reg)
    
    return clf, reg, f"Models trained dynamically on {len(df)} samples (InMemory Mode)."

# Inisialisasi model aktif
clf_model, reg_model, model_source_label = load_or_train_models(df_active)

# ==============================================================================
# NAVIGASI STATE DAN LOGIKA NAVBAR
# ==============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Header Dashboard (Static Positioned, modern, dan sangat stabil)
st.markdown("""
<div class="branding-container">
    <div class="brand-logo">⚡ SENSIROOM AI</div>
    <div class="brand-desc">
        SaaS Workspace Thermal Comfort & Energy Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# Grid Navigasi menggunakan Column Streamlit Asli (mencegah bug tag HTML pecah)
nav_col1, nav_col2, nav_col3 = st.columns(3)
with nav_col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
with nav_col2:
    if st.button("📊 Data", use_container_width=True):
        st.session_state.page = "Data"
with nav_col3:
    if st.button("🔮 Prediction & Insight", use_container_width=True):
        st.session_state.page = "Predict"

st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

# ==============================================================================
# HALAMAN 1: HOME
# ==============================================================================
if st.session_state.page == 'Home':
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.8rem; line-height: 1.2;">
            Revolusi Analisis Kenyamanan & <br><span class="gradient-text">Efisiensi Energi Ruangan</span>
        </h1>
        <p style="font-size: 1.05rem; color: #94a3b8; max-width: 850px; line-height: 1.6; margin-bottom: 0;">
            SensiRoom AI memadukan model fisika kenyamanan termal manusia (ISO 7730 PMV/PPD) 
            dengan algoritma kecerdasan buatan untuk mengoptimalkan ruang kerja dan ruang huni Anda. 
            Dapatkan prediksi presisi untuk konsumsi daya listrik serta wawasan kenyamanan waktu-nyata.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Status Panel
    st.markdown("### 🛠️ Status Dashboard & Keaktifan Model")
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        st.markdown(f"""
        <div class="custom-card" style="border-left: 5px solid #10b981; padding: 1.2rem;">
            <div style="font-weight:700; color:#f8fafc">Database Source:</div>
            <div class="source-badge">{data_source_label}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_status2:
        st.markdown(f"""
        <div class="custom-card" style="border-left: 5px solid #a855f7; padding: 1.2rem;">
            <div style="font-weight:700; color:#f8fafc">Active Core Engine:</div>
            <div class="source-badge" style="background:rgba(59, 130, 246, 0.15); border-color:rgba(59,130,246,0.3); color:#60a5fa">{model_source_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Fitur Utama Sistem")
    
    # Cards layout (Responsive)
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-icon">⚡</div>
            <div class="card-title">Energy Prediction</div>
            <div class="card-desc">
                Prediksikan konsumsi energi HVAC (pemanas/pendingin) dan pencahayaan secara cerdas berdasarkan faktor dimensi ruang, hunian, dan suhu target untuk menekan biaya operasional.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_f2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-icon">🌡️</div>
            <div class="card-title">Thermal Comfort Analytics</div>
            <div class="card-desc">
                Menghitung indeks kenyamanan manusia menggunakan standar dunia <b>PMV (Predicted Mean Vote)</b> dan <b>PPD (Predicted Percentage of Dissatisfied)</b> secara instan.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_f3:
        st.markdown("""
        <div class="custom-card">
            <div class="card-icon">🧠</div>
            <div class="card-title">AI Smart Insights</div>
            <div class="card-desc">
                Sistem rekomendasi dinamis yang memandu Anda untuk mengatur sistem sirkulasi udara, pencahayaan otomatis, serta manajemen hunian cerdas demi penghematan karbon.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Quick Statistics
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📈 Sekilas Parameter Ideal Ruangan")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Suhu Nyaman</div>
            <div class="metric-val">19.5°C - 25.5°C</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Kelembaban Ideal</div>
            <div class="metric-val">40% - 65%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Pencahayaan Kerja</div>
            <div class="metric-val">290 - 500 Lux</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Kebisingan Maksimal</div>
            <div class="metric-val">&lt; 55 dB</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# HALAMAN 2: DATA EXPLORER
# ==============================================================================
elif st.session_state.page == 'Data':
    st.markdown("## 📊 Eksplorasi Data & Analisis Statistik")
    st.write("Jelajahi basis data sensor nyata yang digunakan untuk melatih model kecerdasan buatan SensiRoom.")
    
    # 📈 KEY PERFORMANCE INDICATORS (KPI) SECTION
    st.write("### 🔑 Indikator Kunci Basis Data (KPI)")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    # Menghitung Metrik Secara Dinamis dari df_active
    total_data = len(df_active)
    avg_temp = df_active['temp'].mean()
    avg_energy = df_active['energy_kwh'].mean()
    
    # Rasio kondisi Ideal atau Optimal di database
    nyaman_count = df_active['status'].isin(['Ideal', 'Optimal']).sum()
    rasio_nyaman = (nyaman_count / total_data) * 100 if total_data > 0 else 0
    
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Titik Data</div>
            <div class="metric-val">{total_data:,} <span style="font-size:0.9rem; color:#94a3b8">Sampel</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-Rata Suhu</div>
            <div class="metric-val">{avg_temp:.2f} °C</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-Rata Konsumsi</div>
            <div class="metric-val">{avg_energy:.3f} <span style="font-size:0.9rem; color:#94a3b8">kWh</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tingkat Kenyamanan</div>
            <div class="metric-val">{rasio_nyaman:.1f}% <span style="font-size:0.9rem; color:#10b981">Ideal/Opt</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section Preview Data
    st.write("### 📋 Preview Dataset Smart Room")
    # Menggunakan .head(1000) untuk meminimalkan beban styling pada browser/Streamlit ketika data sangat besar
    st.dataframe(
        df_active.head(1000).style.background_gradient(subset=['temp', 'pmv', 'energy_kwh'], cmap='plasma'),
        use_container_width=True,
        height=320
    )
    
    # Ringkasan Statistik
    with st.expander("📊 Tampilkan Ringkasan Statistik Deskriptif (Describe)", expanded=False):
        st.table(df_active.describe().T)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 📈 Visualisasi Interaktif")
    
    col_g1, col_g2 = st.columns([1.2, 1.8])
    
    # Mapping Warna Unik Sesuai Aturan Dataset Anda
    mapped_colors = {
        'Ideal': '#10b981',        # Hijau
        'Optimalisasi': '#3b82f6', # Biru
        'Peringatan': '#f59e0b',   # Kuning/Orange
        'Kritis': '#ef4444',       # Merah
        'Boros Energi': '#ec4899'  # Pink Soft
    }
    
    with col_g1:
        st.write("#### Distribusi Status Kenyamanan Ruangan")
        
        status_counts = df_active['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Jumlah Ruangan']
        
        fig_hist = px.bar(
            status_counts, 
            x='Status', 
            y='Jumlah Ruangan', 
            color='Status',
            color_discrete_map=mapped_colors,
            template='plotly_dark'
        )
        
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=380,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_g2:
        st.write("#### Korelasi Multi-Dimensi: Suhu vs Konsumsi Energi")
        
        fig_scatter = px.scatter(
            df_active,
            x='temp',
            y='energy_kwh',
            color='status',
            size='occupancy',
            hover_data=['hum', 'lux', 'noise', 'luas'],
            color_discrete_map=mapped_colors,
            labels={
                'temp': 'Suhu (°C)',
                'energy_kwh': 'Energi (kWh)',
                'status': 'Status Ruangan',
                'occupancy': 'Hunian (Orang)'
            },
            template='plotly_dark'
        )
        
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# HALAMAN 3: AI PREDICTION
# ==============================================================================
elif st.session_state.page == 'Predict':
    st.markdown("## 🔮 Dashboard Prediksi & Optimasi Ruangan")
    st.write("Masukkan parameter ruangan Anda saat ini untuk memproyeksikan konsumsi energi serta menghitung indeks kenyamanan secara instan.")
    
    # Form Input User sesuai Fitur Latih Model Anda
    with st.form("input_form"):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        with col_in1:
            st.markdown("##### 👥 Kepadatan Ruang")
            occupancy = st.slider("Jumlah Orang (Occupancy)", min_value=0, max_value=100, step=1, key="input_occupancy")
            
            st.markdown("##### 📏 Dimensi Fisik")
            luas_ruang = st.slider("Luas Ruangan (m²)", min_value=20, max_value=250, step=1, key="input_luas")
            
        with col_in2:
            st.markdown("##### 🌡️ Sensor Termal")
            temperature = st.slider("Suhu Udara (°C)", min_value=15.0, max_value=35.0, step=0.1, key="input_temp")
            humidity = st.slider("Kelembaban Relatif (%)", min_value=15.0, max_value=95.0, step=0.1, key="input_hum")
            
        with col_in3:
            st.markdown("##### 💡 Sensor Lingkungan")
            lux = st.slider("Tingkat Cahaya (Lux)", min_value=10, max_value=1000, step=5, key="input_lux")
            noise = st.slider("Tingkat Kebisingan (dB)", min_value=25.0, max_value=90.0, step=0.5, key="input_noise")
            
        st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Analisis Parameter Ruang")
        st.markdown("</div>", unsafe_allow_html=True)

    # PEMROSESAN PREDIKSI MENGGUNAKAN MODEL MODEL PIPELINE ANDA
    # Siapkan data payload dengan urutan fitur persis saat training: ['occupancy', 'temp', 'hum', 'lux', 'noise', 'luas']
    features_payload = pd.DataFrame({
        'occupancy': [occupancy],
        'temp': [temperature],
        'hum': [humidity],
        'lux': [lux],
        'noise': [noise],
        'luas': [luas_ruang]
    })
    
    # Eksekusi Klasifikasi & Regresi menggunakan model terlatih
    ai_predicted_status = clf_model.predict(features_payload)[0]
    reg_predictions = reg_model.predict(features_payload)[0] # Menghasilkan array [energy_kwh, pmv, ppd]
    
    ai_predicted_energy = reg_predictions[0]
    ai_predicted_pmv = reg_predictions[1]
    ai_predicted_ppd = reg_predictions[2]
    
    # Batasi keluaran logika agar tidak meledak di luar ambang teoretis
    ai_predicted_pmv = np.clip(ai_predicted_pmv, -3.0, 3.0)
    ai_predicted_ppd = np.clip(ai_predicted_ppd, 5.0, 100.0)
    
    # Jika tombol submit ditekan, bersihkan rekomendasi Gemini yang lama agar tidak usang
    if submit_btn:
        st.session_state.gemini_result = None
    
    # Menentukan Warna Status kenyamanan untuk Card Output
    color_map = {
        'Ideal': '#10b981',
        'Optimalisasi': '#3b82f6',
        'Peringatan': '#f59e0b',
        'Kritis': '#ef4444',
        'Boros Energi': '#ec4899'
    }
    status_color = color_map.get(ai_predicted_status, '#f1f5f9')
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 📊 Hasil Proyeksi Parameter")
    
    # Output Card Results
    col_out1, col_out2, col_out3, col_out4 = st.columns(4)
    
    with col_out1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {status_color}">
            <div class="metric-label">Status Kenyamanan</div>
            <div class="metric-val" style="color: {status_color}; font-size: 1.5rem; margin-top: 0.9rem;">
                {ai_predicted_status}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_out2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prediksi Energi</div>
            <div class="metric-val">{ai_predicted_energy:.2f} <span style="font-size:1.1rem; color:#94a3b8">kWh</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_out3:
        pmv_sign = "+" if ai_predicted_pmv > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Indeks PMV</div>
            <div class="metric-val">{pmv_sign}{ai_predicted_pmv:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_out4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Indeks PPD (Ketidakpuasan)</div>
            <div class="metric-val">{ai_predicted_ppd:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 🌡️ PMV VISUAL BAR (DENGAN GLOW EFFECT)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("#### 🌡️ Visualisasi Skala Kenyamanan Termal (PMV)")
    st.write("PMV (Predicted Mean Vote) memproyeksikan sensasi dingin-panas manusia dalam skala -3 (Sangat Dingin) sampai +3 (Sangat Panas).")
    
    # Hitung Persentase Posisi untuk Left CSS
    pmv_percentage = ((ai_predicted_pmv - (-3.0)) / 6.0) * 100
    pmv_percentage = np.clip(pmv_percentage, 0, 100)
    
    st.markdown(f"""
    <div class="pmv-container">
        <div style="display: flex; justify-content: space-between; font-weight:700; font-size:0.9rem; margin-bottom: 0.5rem;">
            <span style="color:#3b82f6">❄️ Dingin (-3)</span>
            <span style="color:#10b981">🍀 Nyaman (0)</span>
            <span style="color:#ef4444">🔥 Panas (+3)</span>
        </div>
        <div class="pmv-track">
            <div class="pmv-indicator" style="left: {pmv_percentage}%;"></div>
        </div>
        <div class="pmv-labels">
            <span>-3</span>
            <span>-2</span>
            <span>-1</span>
            <span>0</span>
            <span>+1</span>
            <span>+2</span>
            <span>+3</span>
        </div>
        <div style="text-align: center; margin-top: 1.2rem; font-weight:700; font-size: 1rem; color:#c084fc">
            Posisi Indeks Saat Ini: {pmv_sign}{ai_predicted_pmv:.2f} ({ai_predicted_status})
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 🧠 AI INSIGHTS BERDASARKAN ATURAN DARI PIPELINE PREP ANDA
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### Rekomendasi & Insight Otomatis")
    
    # Sesi Klasik (Rule-based Insights)
    insights = []
    
    # 1. Analisis Kepadatan Ruangan & Energi
    if occupancy == 0:
        if (temperature <= 23.5) and (lux > 150):
            insights.append("⚠️ <b>TERDETEKSI PEMBOROSAN ENERGI:</b> Ruangan dalam keadaan kosong, namun sistem mendeteksi AC menyala dingin (≤23.5°C) dan lampu menyala terang (>150 Lux). Matikan perangkat HVAC dan pencahayaan segera.")
        elif (temperature > 23.5) and (lux <= 150):
            insights.append("🍀 <b>KONDISI KOSONG IDEAL:</b> Ruangan kosong dengan sistem hemat daya aktif (suhu hangat dan pencahayaan minim). Standar manajemen energi yang sangat baik.")
    
    # 2. Kasus Berpenghuni tapi Gelap
    if (occupancy > 0) and (lux < 290):
        insights.append(f"⚠️ <b>WARNING (Pencahayaan Buruk):</b> Terdapat {occupancy} orang di dalam ruangan, namun tingkat cahaya berada di bawah standar standar kenyamanan visual (< 290 Lux). Harap nyalakan pencahayaan tambahan.")
        
    # 3. Kategori Kapasitas Ruang
    if 26 <= occupancy <= 30:
        insights.append(f"👥 <b>Peringatan Level 1 (Kepadatan Sedang):</b> Ruangan berisi {occupancy} orang. Rentang temperatur disarankan dinaikkan bertahap untuk mengimbangi emisi panas tubuh manusia.")
    elif 31 <= occupancy <= 60:
        insights.append(f"👥 <b>Peringatan Level 2 (Kepadatan Tinggi):</b> Ruangan sangat padat ({occupancy} orang). Sistem tata udara wajib diatur ke kapasitas maksimum guna menghindari pengap.")
    elif occupancy > 60:
        insights.append(f"🚨 <b>KONDISI KRITIS (Overcrowded):</b> Jumlah orang melebihi batas aman sirkulasi sehat (>60 orang). Segera batasi akses masuk and buka ventilasi darurat.")

    # 4. Analisis Parameter Sensor
    if temperature > 26.5:
        insights.append(f"🌡️ <b>Suhu Udara Panas ({temperature}°C):</b> Menimbulkan resiko ketidakpuasan termal. Turunkan setpoint temperatur unit HVAC.")
    elif temperature < 19.5:
        insights.append(f"🌡️ <b>Suhu Udara Dingin ({temperature}°C):</b> Indeks dingin berlebihan terdeteksi. Naikkan setpoint unit termal demi memangkas daya listrik.")

    if humidity > 65:
        insights.append(f"💧 <b>Kelembaban Tinggi ({humidity}%):</b> Menstimulasi kelembaban basah tidak nyaman dan jamur. Aktifkan mode dry-mode pada AC.")
    elif humidity < 40:
        insights.append(f"💧 <b>Kelembaban Terlalu Kering ({humidity}%):</b> Udara kering berpotensi memicu iritasi tenggorokan. Humidifier disarankan dinyalakan.")

    if noise > 55:
        insights.append(f"🔊 <b>Tingkat Bising Berlebih ({noise} dB):</b> Menurunkan konsentrasi kerja penghuni. Cari sumber polusi suara akustik Anda.")

    # 5. Summary Efisiensi Energi
    if ai_predicted_energy > 0.60:
        insights.append(f"🔌 <b>Beban Daya Tinggi ({ai_predicted_energy:.2f} kWh):</b> Sistem bekerja ekstra keras. Batasi beban hunian atau atur mode penjadwalan otomatis.")
    else:
        insights.append(f"🔌 <b>Beban Daya Efisien ({ai_predicted_energy:.2f} kWh):</b> Efisiensi energi ruangan terpelihara dengan sangat baik.")

    # Render Rule-based Insights
    insights_html = "".join([f'<li class="insight-item">{ins}</li>' for ins in insights])
    
    st.markdown(f"""
    <div class="custom-card" style="border-left: 5px solid #a855f7;">
        <div style="font-size: 1.2rem; font-weight:700; margin-bottom: 1rem; display:flex; align-items:center; gap:0.5rem">
            <span>💡</span> Wawasan Standar Pengaturan Ruang (Rule-Based)
        </div>
        <ul class="insight-list">
            {insights_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 🧠 PROMPT & GENERATOR DEEP INSIGHTS GEMINI (KONDISIONAL)
    if gemini_api_key:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("### 🧠 Generative AI Insights (Gemini)")
        st.write("Dapatkan analisis mendalam dan saran operasional cerdas yang dipersonalisasi secara dinamis menggunakan model AI terkanggih.")
        
        # Skenario Prompting untuk Gemini
        system_instruction = (
            "Anda adalah asisten AI ahli energi dan kenyamanan termal ruangan (smart building consultant). "
            "Tugas Anda adalah memberikan rekomendasi aksi praktis, cerdas, dan profesional berdasarkan data sensor yang diberikan. "
            "Gunakan bahasa Indonesia yang profesional, ringkas, persuasif, dan mudah dimengerti. "
            "Fokus pada efisiensi energi (kWh) dan kenyamanan termal manusia (skala PMV & PPD)."
        )
        
        prompt_payload = (
            f"Analisis kondisi ruangan berikut secara detail:\n"
            f"- Jumlah Penghuni saat ini: {occupancy} orang\n"
            f"- Dimensi Luas Ruangan: {luas_ruang} m²\n"
            f"- Suhu Udara Terbaca: {temperature:.1f}°C\n"
            f"- Kelembaban Relatif: {humidity:.1f}%\n"
            f"- Kecerahan Lampu: {lux} Lux\n"
            f"- Tingkat Polusi Suara: {noise:.1f} dB\n\n"
            f"Hasil Kalkulasi Prediksi Model Machine Learning SensiRoom:\n"
            f"- Kategori Status Ruang: {ai_predicted_status}\n"
            f"- Proyeksi Konsumsi Energi Listrik: {ai_predicted_energy:.2f} kWh\n"
            f"- Nilai Indeks Kenyamanan PMV: {ai_predicted_pmv:.2f}\n"
            f"- Estimasi Persentase Ketidakpuasan (PPD): {ai_predicted_ppd:.1f}%\n\n"
            f"Berikan analisis mendalam dan poin-poin saran strategis untuk menghemat energi sekaligus mempertahankan tingkat kepuasan termal ideal penghuni."
        )

        if st.button("✨ Generasikan AI Insights Mendalam (Gemini 2.5)", type="primary"):
            with st.spinner("Menganalisis skenario ruang Anda dengan Gemini AI..."):
                gemini_text_result = get_gemini_insights(prompt_payload, system_instruction, gemini_api_key)
                st.session_state.gemini_result = gemini_text_result
                
        # Jika rekomendasi Gemini ada di memori sesi, render langsung di dalam kartu bermotif SaaS
        if st.session_state.gemini_result:
            st.markdown(f"""
            <div class="custom-card" style="border-left: 5px solid #ec4899; margin-top: 1.5rem; padding: 1.8rem;">
                <div style="font-size: 1.2rem; font-weight:700; margin-bottom: 1rem; display:flex; align-items:center; gap:0.5rem; color: #f472b6;">
                    <span>✨</span> SensiRoom AI Generative Insights (Gemini)
                </div>
                <div style="color: #cbd5e1; line-height: 1.6; font-size: 0.95rem;">
{st.session_state.gemini_result}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Tips Portofolio:** Masukkan **Gemini API Key** Anda pada menu input di panel kiri (sidebar) untuk membuka fitur rekomendasi prediktif berbasis Generative AI secara interaktif!")

    # ==============================================================================
    # TOMBOL RESET PARAMETER DAN REKOMENDASI (DITEMPATKAN DI BAGIAN PALING BAWAH)
    # ==============================================================================
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08)'><br>", unsafe_allow_html=True)
    
    # Tampilkan banner sukses jika baru saja di-reset
    if st.session_state.get("show_reset_success", False):
        st.success("🎉 Semua parameter dan rekomendasi berhasil di-reset ke nilai default!")
        st.session_state.show_reset_success = False

    col_reset1, col_reset2, col_reset3 = st.columns([4, 2.5, 4])
    with col_reset2:
        st.button(
            "🔄 Reset Semua Parameter & Rekomendasi", 
            use_container_width=True, 
            type="secondary",
            on_click=reset_all_parameters
        )

# Footer Portfolio info
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 0.8rem; color: #475569; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 2rem;">
    SensiRoom AI Dashboard • Dikembangkan sebagai Portfolio Data Science & IoT Tingkat Lanjut. <br>
    © 2026 Seluruh hak cipta dilindungi.
</div>
""", unsafe_allow_html=True)