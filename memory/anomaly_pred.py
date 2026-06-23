import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Read from the prediction results
data = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\results\usage_pred_results.csv")

# Drop any potential NaNs or infinite values (created by pct_change)
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Using predicted_forecast instead of memory_usage_pct
features = [ "predicted_forecast", "predicted_rolling_std_24h", "predicted_growth_rate", "predicted_acceleration"]

X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    contamination="auto",
    random_state=42
)

model.fit(X_scaled)

scores = model.decision_function(X_scaled)

data["anomaly_score"] = -scores

# Parse timezone safely. If it's already naive, skip tz_localize to avoid TypeError
data["ts"] = pd.to_datetime(data["ts"], format='mixed')
if data["ts"].dt.tz is not None:
    data["ts"] = data["ts"].dt.tz_localize(None)

# Save result
output_path = r"C:\AI-Predictive-Maintenance\memory\results\anomaly_pred_result.csv"
data.to_csv(output_path, index=False)

print(f"Anomaly prediction results saved to {output_path}")
