import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_real = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\results\result_riskscoring.csv")
df_predicted = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\results\result_predicted_riskscoring.csv")

# Convert timestamp to datetime
df_real['ts'] = pd.to_datetime(df_real['ts'])
df_predicted['ts'] = pd.to_datetime(df_predicted['ts'])

# Sort by timestamp
df_real = df_real.sort_values('ts')
df_predicted = df_predicted.sort_values('ts')

# Set seaborn style
sns.set_theme(style="whitegrid")

# Ensure results directory exists for output images
output_dir = r"C:\AI-Predictive-Maintenance\memory\results"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. Plot Memory Usage with Risk Coloring
# ---------------------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(df_real['ts'], df_real['memory_usage_pct'], color='lightgrey', alpha=0.5, label='Actual Usage Line')
plt.plot(df_predicted['ts'], df_predicted['predicted_forecast'], color='lightblue', alpha=0.5, linestyle='--', label='Predicted Usage Line')

risk_colors = {
    'healthy': 'green',
    'low risk': 'gold',
    'med risk': 'orange',
    'high risk': 'red',
    'critical': 'darkred'
}

for status, color in risk_colors.items():
    subset_real = df_real[df_real['status_risk_score'] == status]
    if not subset_real.empty:
        plt.scatter(subset_real['ts'], subset_real['memory_usage_pct'], color=color, label=f'Actual: {status}', s=15, zorder=5)
        
    subset_pred = df_predicted[df_predicted['status_risk_score'] == status]
    if not subset_pred.empty:
        plt.scatter(subset_pred['ts'], subset_pred['predicted_forecast'], color=color, marker='x', label=f'Predicted: {status}', s=20, zorder=6)

plt.title('Actual vs Predicted Memory Usage with Risk Coloring')
plt.xlabel('Time')
plt.ylabel('Memory Usage (%)')

# Deduplicate legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'))
plt.show()

# ---------------------------------------------------------
# 2. Plot Risk Scores
# ---------------------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(df_real['ts'], df_real['risk_score'], label='Actual Risk Score', color='blue', alpha=0.7)
plt.plot(df_predicted['ts'], df_predicted['risk_score'], label='Predicted Risk Score', color='red', alpha=0.7, linestyle='--')
plt.title('Actual vs Predicted Risk Score Over Time')
plt.xlabel('Time')
plt.ylabel('Risk Score')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'risk_score_comparison.png'))
plt.show()

# ---------------------------------------------------------
# 3. Plot Anomaly Scores
# ---------------------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(df_real['ts'], df_real['anomaly_score'], label='Actual Anomaly Score', color='green', alpha=0.7)
plt.plot(df_predicted['ts'], df_predicted['anomaly_score'], label='Predicted Anomaly Score', color='orange', alpha=0.7, linestyle='--')
plt.title('Actual vs Predicted Anomaly Score Over Time')
plt.xlabel('Time')
plt.ylabel('Anomaly Score')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'anomaly_score_comparison.png'))
plt.show()

print(f"Plots saved successfully in {output_dir}")
