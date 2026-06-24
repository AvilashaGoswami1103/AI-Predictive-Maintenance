import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    """
    Trains and infers anomalies using Isolation Forest on specified features.
    """
    data = data.copy()
    
    # We must handle NaNs for Isolation Forest
    # Drop rows where critical features are NaN, usually resulting from rolling windows
    features = ["memory_usage_pct", "rolling_std_24h", "growth_rate", "acceleration"]
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
    
    # Merge back to original dataframe to maintain size/index
    data["anomaly_score"] = np.nan
    data.loc[data_clean.index, "anomaly_score"] = data_clean["anomaly_score"]
    
    return data
