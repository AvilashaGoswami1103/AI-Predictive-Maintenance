import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\out.csv")

# Sort by time
df['ts'] = pd.to_datetime(df['ts'], format='mixed', utc=True).dt.tz_convert(None)
df = df.sort_values('ts')

# Shift target leakage columns by 1 (per host) to use historical data for current prediction
leakage_cols = ['rolling_mean_1h', 'rolling_mean_24h', 'rolling_std_1h', 'rolling_std_24h', 
                'growth_rate', 'acceleration', 'Z_score', 'trend', 'volatility_ratio']
df[leakage_cols] = df.groupby('host_id')[leakage_cols].shift(1)

# Drop columns that are not features (e.g., id, ts, status) and separate target
features = df.drop(columns=['id', 'ts', 'status', 'memory_usage_pct'])
target = df['memory_usage_pct']

# Split data: 75% train, 10% val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(features, target, test_size=0.15, shuffle=False)

# Next, split temp into train (75% of total) and val (10% of total)
# 10 / 85 = 0.117647...
val_size = 0.10/ 0.85
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, shuffle=False)

print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Initialize XGBoost Regressor
model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.6, random_state=42)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

print(f"Train RMSE: {mean_squared_error(y_train, y_pred_train) ** 0.5:.4f}")
print(f"Val RMSE: {mean_squared_error(y_val, y_pred_val) ** 0.5:.4f}")
print(f"Test RMSE: {mean_squared_error(y_test, y_pred_test) ** 0.5:.4f}")

# Plotting: Scatter plots
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_val, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual Memory Usage %')
plt.ylabel('Predicted Memory Usage %')
plt.title('Validation Set: Actual vs Predicted')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='green', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual Memory Usage %')
plt.ylabel('Predicted Memory Usage %')
plt.title('Test Set: Actual vs Predicted')
plt.legend()

plt.tight_layout()
plt.savefig(r'C:\AI-Predictive-Maintenance\memory\prediction_scatter.png')

# Plotting: Time series line plots for the test set
plt.figure(figsize=(15, 6))
plt.plot(df.iloc[X_test.index]['ts'], y_test, label='Actual', alpha=0.7)
plt.plot(df.iloc[X_test.index]['ts'], y_pred_test, label='Predicted', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Memory Usage %')
plt.title('Test Set: Actual vs Predicted over Time')
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\AI-Predictive-Maintenance\memory\prediction_timeseries.png')

print("Plots saved successfully.")
