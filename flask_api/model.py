import joblib

def load_model(model_path, scaler_path):
    model = joblib.load(model_path) 
    scaler = joblib.load(scaler_path)  
    return model, scaler