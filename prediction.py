import pandas as pd
from model import load_model 

# --- Setelan Fitur yang Sama Persis dengan Training ---
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
    "Lainnya": 'job_type_manager' 
}


def predict_salary(experience, skill_count, job_type_selected):
    """
    Mengambil input user, mengubahnya menjadi format fitur yang benar, 
    dan melakukan prediksi gaji. Menggunakan try...except untuk robustness.
    """
    
    # === PENANGANAN ERROR (try...except) DIMULAI ===
    try:
        model = load_model(MODEL_FILENAME)
        
        if model is None:
            # Mengembalikan string error jika load_model gagal memuat file
            return "Model gagal dimuat. Pastikan file model ada dan benar."

        # 1. Inisialisasi DataFrame dengan nilai 0 untuk semua kolom OHE
        data = {feature: 0 for feature in FEATURES}
        
        # 2. Isi nilai dari input user
        data['experience_required'] = experience
        data['skill_count'] = skill_count
        
        # 3. Set 1 untuk kolom OHE yang dipilih user
        ohe_col = JOB_TYPE_MAP.get(job_type_selected, 'job_type_manager') 
        if ohe_col in data:
            data[ohe_col] = 1
        
        # 4. Konversi ke DataFrame dan pastikan urutan kolomnya sama persis
        input_df = pd.DataFrame(data, index=[0])
        input_df = input_df[FEATURES]
        
        # 5. Prediksi
        prediction = model.predict(input_df)[0]
        
        return prediction

    except Exception as e:
        # Menangkap error lain (misalnya, error saat prediksi/transformasi data)
        return f"Terjadi error saat proses prediksi: {e}"
