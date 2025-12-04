import pandas as pd
from model import load_model # Memuat model dari script modelrandom_forest_regressor_model.py

# menggunakan Setelan Fitur yang Sama Persis dengan Week 13 yang GUIDED
MODEL_FILENAME = 'random_forest_regressor_model.pkl'
OHE_JOB_TYPES = [
    'job_type_Full time', 'job_type_Full-time', 'job_type_Internship', 
    'job_type_Part-time', 'job_type_Remote', 'job_type_Working student', 
    'job_type_berufseinstieg', 'job_type_berufserfahren', 'job_type_manager', 
    'job_type_professional / experienced'
]
BASE_FEATURES = ['experience_required', 'skill_count']
FEATURES = BASE_FEATURES + OHE_JOB_TYPES

# Mapping input user ke kolom OHE
JOB_TYPE_MAP = {
    "Full-time": 'job_type_Full-time',
    "Part-time": 'job_type_Part-time',
    "Remote": 'job_type_Remote',
    "Internship": 'job_type_Internship',
    # saya menambahkan mapping yang lain jika ingin memasukkan ke 'Lainnya'
    "Lainnya": 'job_type_manager' 
}


def predict_salary(experience, skill_count, job_type_selected):
    """
    Mengambil input user, mengubahnya menjadi format fitur yang benar, 
    dan melakukan prediksi gaji.
    """
    model = load_model(MODEL_FILENAME)
    if model is None:
        return "Model not loaded. Check model file."

    # 1. Inisialisasi DataFrame dengan nilai 0 untuk semua kolom OHE
    data = {feature: 0 for feature in FEATURES}
    
    # 2. Isi nilai dari input user
    data['experience_required'] = experience
    data['skill_count'] = skill_count
    
    # 3. Set 1 untuk kolom OHE yang dipilih user
    # menggunakan mapping untuk mendapatkan nama kolom OHE
    ohe_col = JOB_TYPE_MAP.get(job_type_selected, 'job_type_manager') # menjadi Default ke manager jika tidak ditemukan
    if ohe_col in data:
        data[ohe_col] = 1
    
    # 4. mengkonversi menjadi ke DataFrame dan pastikan urutan kolomnya sama persis
    input_df = pd.DataFrame(data, index=[0])
    input_df = input_df[FEATURES]
    
    # 5. Prediksi
    prediction = model.predict(input_df)[0]
    
    return prediction