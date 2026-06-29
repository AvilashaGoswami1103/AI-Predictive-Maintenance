import pandas as pd
import numpy as np
import joblib
import os

def predict_anomalies_inference(data):
    """
    Detects anomalies on predicted forecasts using the pre-trained Isolation Forest model.
    """
    data = data.copy()
    
    features = ["predicted_forecast", "predicted_rolling_std_24h", "predicted_growth_rate", "predicted_acceleration"]
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    if data_clean.empty:
        print("Data empty after dropping NaNs in anomaly prediction features.")
        return data
        
    X = data_clean[features]
    
    # Load model and scaler
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    scaler_path = os.path.join(model_dir, "anomaly_pred_scaler.pkl")
    model_path = os.path.join(model_dir, "anomaly_pred_model.pkl")
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Predicted Anomaly detection models not found in {model_dir}. Please run the training pipeline first.")
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    X_scaled = scaler.transform(X)
    scores = model.decision_function(X_scaled)
    
    data_clean["anomaly_score"] = -scores
    
    # Apply thresholds (identical to training logic)
    data_clean.loc[data_clean["anomaly_score"] <= 0.22, "anomaly_score"] = -0.3
    data_clean.loc[data_clean["predicted_growth_rate"].abs() >= 0.5, "anomaly_score"] = 0.3
    
    data["anomaly_score"] = np.nan
    data.loc[data_clean.index, "anomaly_score"] = data_clean["anomaly_score"]
    
    return data
