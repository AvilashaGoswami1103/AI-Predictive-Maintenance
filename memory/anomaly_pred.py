import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def predict_anomalies(data):
    """
    Detects anomalies on predicted forecasts using Isolation Forest.
    """
    data = data.copy()
    
    # Drop rows where critical predicted features are NaN
    features = ["predicted_forecast", "predicted_rolling_std_24h", "predicted_growth_rate", "predicted_acceleration"]
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    if data_clean.empty:
        return data
        
    X = data_clean[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination="auto",
        random_state=42
    )
    
    model.fit(X_scaled)
    scores = model.decision_function(X_scaled)
    
    data_clean["anomaly_score"] = -scores
    
    # Merge back to original data
    data["anomaly_score"] = np.nan
    data.loc[data_clean.index, "anomaly_score"] = data_clean["anomaly_score"]
    
    return data
