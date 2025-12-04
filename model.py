import joblib

def load_model(filename='random_forest_regressor_model.pkl'):
    """Memuat model yang sudah dilatih dari file .pkl"""
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        print(f"Error: File model '{filename}' tidak ditemukan.")
        return None
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return None
