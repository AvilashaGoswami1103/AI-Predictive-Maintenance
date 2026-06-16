import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

data = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\out.csv")

data.dropna()

features = [[ "memory_usage_pct", "rolling_std_24h", "growth_rate", "acceleration"]]

X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    contamination = "auto",
    random_state=42
)

model.fit(X_scaled)

scores = model.decision_function(X_scaled)

data["anomaly_score"] = -scores

data["ts"] = pd.to_datetime(data["ts"], format='mixed').dt.tz_localize(None)
data.to_excel("anomaly_result.xlsx")

