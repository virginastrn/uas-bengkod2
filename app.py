import os
import streamlit as st
import joblib

# Jalur Absolut
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'churn_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

st.title('🔍 Detektif Model AI')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    st.success("✅ File model dan scaler berhasil dibaca!")
    
    st.subheader("Rahasia Model Terungkap:")
    st.write(f"Model ini menuntut **{scaler.n_features_in_}** fitur/kolom data.")
    
    if hasattr(scaler, 'feature_names_in_'):
        st.write("Ini dia daftar nama kolom aslinya (Copy bagian ini):")
        st.code(list(scaler.feature_names_in_))
    else:
        st.warning("Model tidak menyimpan nama kolom, hanya jumlahnya saja.")
        
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")