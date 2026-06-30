import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Churn Pelanggan", page_icon="📊")

# 1. Jalur Absolut (Wajib agar tidak error FileNotFoundError di server)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'churn_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# 2. Load model dan scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error(f"Error: File model tidak ditemukan di lokasi {BASE_DIR}")
    st.stop()

# Judul
st.title('📊 Prediksi Churn Pelanggan')
st.markdown('Aplikasi ini memprediksi apakah pelanggan akan churn (berhenti) atau tetap berlangganan menggunakan Machine Learning.')
st.divider()

# 3. Form Input dengan Layout 2 Kolom agar rapi
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Demografi")
    age = st.number_input('Usia', min_value=15, max_value=100, value=25)
    
    gender_input = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    gender_encoded = 1 if gender_input == "Laki-laki" else 0
    
    sub_input = st.selectbox("Tipe Pelanggan", ["Basic", "Premium"])
    sub_encoded = 1 if sub_input == "Premium" else 0

with col2:
    st.subheader("Aktivitas & Tagihan")
    total_spent = st.number_input('Total Pengeluaran ($)', min_value=0.0, value=150.0)
    avg_order_value = st.number_input('Tagihan Bulanan ($)', min_value=0.0, value=20.0)
    total_visits = st.number_input('Total Penggunaan Layanan (Kunjungan)', min_value=0, value=50)
    delivery_delay_days = st.number_input('Masa Aktif / Keterlambatan (Hari)', min_value=0, value=12)

st.divider()

# 4. Tombol Eksekusi
if st.button('🚀 Prediksi Churn', use_container_width=True):
    
    # Feature Engineering Otomatis (sesuai yang kita buat di Colab)
    spent_per_visit = total_spent / (total_visits + 1)
    
    # Susun input menjadi array SESUAI URUTAN FITUR saat di Colab
    input_data = np.array([[
        age, 
        gender_encoded, 
        sub_encoded, 
        total_spent, 
        avg_order_value, 
        total_visits, 
        delivery_delay_days,
        spent_per_visit
    ]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    probabilitas = model.predict_proba(input_scaled)[0]
    
    # Menampilkan Hasil Utama
    st.markdown("### Hasil Prediksi:")
    if prediction[0] == 1:
        st.error('⚠️ **Peringatan!** Pelanggan ini berpotensi **CHURN**.')
    else:
        st.success('✅ **Aman!** Pelanggan ini kemungkinan akan **TETAP BERLANGGANAN**.')

    # 5. Persentase Keyakinan Model (Tanpa Grafik)
    st.markdown("#### Persentase Keyakinan Model:")
    col_a, col_b = st.columns(2)
    col_a.metric("Tetap Berlangganan", f"{probabilitas[0] * 100:.2f}%")
    col_b.metric("Potensi Churn", f"{probabilitas[1] * 100:.2f}%")