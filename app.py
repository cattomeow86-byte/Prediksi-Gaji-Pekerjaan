import streamlit as st
import pandas as pd
from prediction import predict_salary # Memanggil fungsi prediksi dari prediction.py

# --- bagian Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Gaji Pekerjaan",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('💸 Prediksi Gaji Pekerjaan')
st.markdown('### Menggunakan Random Forest Regressor')
st.write('Aplikasi untuk memprediksi rata-rata gaji berdasarkan fitur pekerjaan.')

# --- bagian Input Pengguna di Sidebar ---
st.sidebar.header('⚙️ Atur Fitur Pekerjaan')

# Input Experience
experience = st.sidebar.slider('Pengalaman yang Dibutuhkan (Tahun)', 1, 15, 5)

# kode untuk mengnput Skill Count
skill_count = st.sidebar.slider('Perkiraan Jumlah Skill yang Relevan', 1, 10, 5)

# Input Job Type (menggunakan keys dari mapping di prediction.py)
JOB_TYPE_OPTIONS = ["Full-time", "Part-time", "Remote", "Internship", "Lainnya"]
job_type_selected = st.sidebar.selectbox('Tipe Pekerjaan', JOB_TYPE_OPTIONS)

# --- Tombol untuk Prediksi Gaji ---

if st.button('Hitung Prediksi Gaji'):
    
    # untuk memanggil fungsi prediksi
    prediction = predict_salary(experience, skill_count, job_type_selected)
    
    st.subheader('💸 Hasil Prediksi Rata-rata Gaji')

    if isinstance(prediction, str):
        st.error(prediction) # Menampilkan error jika model gagal dimuat
    else:
        # Format output menjadi mata uang Rupiah
        harga_formatted = f"Rp {prediction:,.2f}".replace(",", "_TEMP_").replace(".", ",").replace("_TEMP_", ".")
        st.success(f"Rata-rata Gaji yang Diprediksi: **{harga_formatted}**")

# --- Keterangan ---
st.markdown('---')
st.caption("Model: Random Forest Regressor")