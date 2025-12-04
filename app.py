import streamlit as st
import joblib
from prediction import predict_salary 

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Gaji Pekerjaan",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('💰 Prediksi Gaji Pekerjaan')
st.markdown('### Menggunakan Random Forest Regressor')
st.write('Aplikasi untuk memprediksi rata-rata gaji berdasarkan fitur pekerjaan.')

# --- Input Pengguna di Sidebar ---
st.sidebar.header('⚙️ Atur Fitur Pekerjaan')
experience = st.sidebar.slider('Pengalaman yang Dibutuhkan (Tahun)', 1, 15, 5)
skill_count = st.sidebar.slider('Perkiraan Jumlah Skill yang Relevan', 1, 10, 5)
JOB_TYPE_OPTIONS = ["Full-time", "Part-time", "Remote", "Internship", "Lainnya"]
job_type_selected = st.sidebar.selectbox('Tipe Pekerjaan', JOB_TYPE_OPTIONS)

# --- Tombol Prediksi ---

if st.button('Hitung Prediksi Gaji 🚀'):
    
    prediction = predict_salary(experience, skill_count, job_type_selected)
    
    st.subheader('💸 Hasil Prediksi Rata-rata Gaji')

    if isinstance(prediction, str):
        # Jika hasilnya string, berarti itu adalah pesan error
        st.error(prediction) 
    else:
        # Jika hasilnya float/int, itu adalah hasil prediksi yang sukses
        # Format output menjadi mata uang Rupiah (contoh sederhana)
        harga_formatted = f"Rp {prediction:,.2f}".replace(",", "_TEMP_").replace(".", ",").replace("_TEMP_", ".")
        st.success(f"Rata-rata Gaji yang Diprediksi: **{harga_formatted}**")

# --- Keterangan ---
st.markdown('---')
st.caption("Model: Random Forest Regressor")
