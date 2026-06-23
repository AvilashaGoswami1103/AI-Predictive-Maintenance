import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Load data
df = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\processed_ip_data\out.csv")

# Sort by time
df['ts'] = pd.to_datetime(df['ts'], format='mixed', utc=True).dt.tz_convert(None)
df = df.sort_values('ts')
# Preserve unshifted columns for the final results CSV
df['orig_rolling_std_24h'] = df['rolling_std_24h']
df['orig_growth_rate'] = df['growth_rate']
df['orig_acceleration'] = df['acceleration']

# Shift target leakage columns by 1 (per host) to use historical data for current prediction
leakage_cols = ['rolling_mean_1h', 'rolling_mean_24h', 'rolling_std_1h', 'rolling_std_24h', 
                'growth_rate', 'acceleration', 'Z_score', 'trend', 'volatility_ratio']
df[leakage_cols] = df.groupby('host_id')[leakage_cols].shift(1)

# Setup Forecasting Horizon
FORECAST_HORIZON = 30
df['target_forecast'] = df.groupby('host_id')['memory_usage_pct'].shift(-FORECAST_HORIZON)

# Remove Invalid Rows resulting from shifts
df = df.dropna()

# Drop columns that are not features (e.g., id, ts, status, target_forecast) and separate target
# Note: memory_usage_pct is kept as a feature for forecasting
features = df.drop(columns=['id', 'ts', 'status', 'host_id', 'target_forecast', 'orig_rolling_std_24h', 'orig_growth_rate', 'orig_acceleration'])
target = df['target_forecast']

# Split data: 75% train, 10% val, 15% test
# First, split into temp (85%) and test (15%)
# We use shuffle=False to keep the time series sequence
X_temp, X_test, y_temp, y_test = train_test_split(features, target, test_size=0.15, shuffle=False)

# Next, split temp into train (75% of total) and val (10% of total)
# 10 / 85 = 0.117647...
val_size = 0.10/ 0.85
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, shuffle=False)

print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Initialize XGBoost Regressor for forecasting
model = xgb.XGBRegressor(
    n_estimators=2000, 
    max_depth=8, 
    learning_rate=0.05, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)

# Train the model
print("Training XGBoost model...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

def print_metrics(name, y_true, y_pred):
    print(f"--- {name} Metrics ---")
    print(f"RMSE: {mean_squared_error(y_true, y_pred) ** 0.5:.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R2:   {r2_score(y_true, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.4f}\n")

print_metrics("Train", y_train, y_pred_train)
print_metrics("Validation", y_val, y_pred_val)
print_metrics("Test", y_test, y_pred_test)

# Save test results to CSV
results_df = df.loc[X_test.index, ['id', 'ts', 'host_id', 'memory_usage_pct', 'orig_rolling_std_24h', 'orig_growth_rate', 'orig_acceleration']].copy()
results_df = results_df.rename(columns={
    'orig_rolling_std_24h': 'rolling_std_24h',
    'orig_growth_rate': 'growth_rate',
    'orig_acceleration': 'acceleration'
})
results_df['actual_forecast'] = y_test
results_df['predicted_forecast'] = y_pred_test

# Set datetime index to allow time-based rolling windows, grouped by host
results_df = results_df.sort_values('ts').set_index('ts')

# Calculate predicted stats properly
results_df['predicted_rolling_std_24h'] = results_df.groupby('host_id')['predicted_forecast'].rolling('24h').std().reset_index(level=0, drop=True)
results_df['predicted_growth_rate'] = results_df.groupby('host_id')['predicted_forecast'].pct_change()
results_df['predicted_acceleration'] = results_df.groupby('host_id')['predicted_growth_rate'].diff()

# Reset index to bring 'ts' back as a normal column before saving
results_df = results_df.reset_index()
results_path = r'C:\AI-Predictive-Maintenance\memory\results\usage_pred_results.csv'
results_df.to_csv(results_path, index=False)
print(f"Test results saved successfully to {results_path}")


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
plt.savefig(r'C:\AI-Predictive-Maintenance\memory\results\forecast_scatter.png')

# Plotting: Time series line plots for the test set
test_hosts = df.loc[X_test.index, 'host_id']
top_host = test_hosts.value_counts().idxmax() if not test_hosts.empty else None

plt.figure(figsize=(15, 6))
if top_host is not None:
    plot_mask_vals = (test_hosts == top_host).values
    plt.plot(df.loc[X_test.index[plot_mask_vals], 'ts'], y_test.iloc[plot_mask_vals], label='Actual', alpha=0.7)
    plt.plot(df.loc[X_test.index[plot_mask_vals], 'ts'], y_pred_test[plot_mask_vals], label='Predicted', alpha=0.7)
    plt.title(f'Test Set: Actual vs Predicted over Time - Host {top_host}')
else:
    plt.plot(df.loc[X_test.index, 'ts'], y_test, label='Actual', alpha=0.7)
    plt.plot(df.loc[X_test.index, 'ts'], y_pred_test, label='Predicted', alpha=0.7)
    plt.title('Test Set: Actual vs Predicted over Time ')

plt.xlabel('Time')
plt.ylabel('Memory Usage %')
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\AI-Predictive-Maintenance\memory\results\forecast_timeseries.png')

print("Plots saved successfully.")