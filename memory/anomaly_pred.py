import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Read from the prediction results
data = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\results\usage_pred_results.csv")

# Drop any potential NaNs (fixed from anomaly_det.py where it wasn't assigned back)
data = data.dropna()

# Using predicted_forecast instead of memory_usage_pct
features = [ "predicted_forecast", "rolling_std_24h", "growth_rate", "acceleration"]

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

# Handle timezone localization similarly
data["ts"] = pd.to_datetime(data["ts"], format='mixed').dt.tz_localize(None)

# Save result
output_path = r"C:\AI-Predictive-Maintenance\memory\results\anomaly_pred_result.csv"
data.to_csv(output_path, index=False)

print(f"Anomaly prediction results saved to {output_path}")
