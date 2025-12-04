import joblib

def load_model(filename='random_forest_regressor_model.pkl'):
    """Memuat model yang sudah dilatih dari file random_forest_regressor_model.pkl."""
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        return None