import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\results\anomaly_result.csv")
df_pred = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\results\anomaly_pred_result.csv")

df_pred = df_pred.rename(columns={'anomaly_score': 'predicted_anomaly_score'})

# Merge using 'id' to avoid potential timestamp string formatting mismatches between scripts
df_rs = pd.merge(df1, df_pred[['id', 'predicted_forecast', 'predicted_anomaly_score']], on='id', how='inner')

#calculate current risk based on mem usage
def current_risk(memory_usage):
    if memory_usage < 50:
        return 10
    
    elif memory_usage < 70:
        return 30
    
    elif memory_usage < 86:
        return 70
    
    else :
        return 100
    
# calculate forecast risk based on forecast of mem usage pct
def forecast_risk(forecast_memory):
    if forecast_memory < 50:
        return 10
    
    elif forecast_memory < 70:
        return 40
    
    elif forecast_memory < 85:
        return 80
    
    else :
        return 100
    

#calculate the status of risk
def get_risk_status(score):
    if pd.isna(score):
        return 'unknown'
    if score < 30:
        return 'healthy'
    elif score < 60:
        return 'low risk'
    elif score < 76:
        return 'med risk'
    else:
        return 'critical'
    

#calculate anomaly risk based on anomaly score
def anomaly_risk(score, min_score=-0.3, max_score=0.3):

    risk = 100 * ((score- min_score)/ (max_score - min_score))


    return np.clip(risk,0,100)

#calculate the predicted anomaly risk based on the anomaly detection on predicted data
def predicted_anomaly_risk(score, min_score=-0.3, max_score=0.3):

    risk = 100 * ((score - min_score) / (max_score - min_score))

    return np.clip(risk, 0, 100)


max_std = df_rs['rolling_std_24h'].max()
#calculate stability risk based on rolling standard for 24 h
def stability_risk(std):
    if pd.isna(std) or pd.isna(max_std) or max_std == 0:
        return 0
    return min(
        100,
        (std / max_std) * 100
    )

df_rs['risk_score'] = (
    0.15 * df_rs['memory_usage_pct'].apply(current_risk) +
    0.20 * df_rs['predicted_forecast'].apply(forecast_risk) +
    0.20 * df_rs['anomaly_score'].apply(anomaly_risk) +
    0.20 * df_rs['predicted_anomaly_score'].apply(predicted_anomaly_risk) +
    0.25 * df_rs['rolling_std_24h'].apply(stability_risk)
)

df_rs['status_risk_score'] = (
    df_rs['risk_score'].apply(get_risk_status)
)

result_path = r"C:\AI-Predictive-Maintenance\memory\results\result_riskscoring.csv"
df_rs.to_csv(result_path, index=False)

# Continuous line plot over time
if 'ts' in df_rs.columns:
    df_rs = df_rs.sort_values('ts')
    x_col = 'ts'
else:
    x_col = df_rs.index

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

plt.figure(figsize=(14, 6))
ax = plt.gca()

# Prepare x-axis and data
x = np.arange(len(df_rs))
y_mem = df_rs['memory_usage_pct'].values
y_pred = df_rs['predicted_forecast'].values

# Map statuses to colors
status_to_color = {
    'healthy': 'green',
    'low risk': '#ffd700',  # yellow/gold
    'med risk': 'orange',
    'critical': 'red',
    'unknown': 'gray'
}
# Default to blue if somehow a status is missing
colors = df_rs['status_risk_score'].map(lambda s: status_to_color.get(s, 'blue')).tolist()

# Create line segments for memory usage
points_mem = np.array([x, y_mem]).T.reshape(-1, 1, 2)
segments_mem = np.concatenate([points_mem[:-1], points_mem[1:]], axis=1)
lc_mem = LineCollection(segments_mem, colors=colors[:-1], linewidth=2.5)

# Create line segments for forecast
points_pred = np.array([x, y_pred]).T.reshape(-1, 1, 2)
segments_pred = np.concatenate([points_pred[:-1], points_pred[1:]], axis=1)
# Use a dashed line style for forecast to differentiate it
lc_pred = LineCollection(segments_pred, colors=colors[:-1], linewidth=2.5, linestyles='dashed')

ax.add_collection(lc_mem)
ax.add_collection(lc_pred)
ax.autoscale()

# Configure X-axis ticks
if x_col == 'ts':
    step = max(1, len(df_rs) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(df_rs['ts'].iloc[::step], rotation=45)

# Create a custom legend
legend_elements = [
    Line2D([0], [0], color='black', lw=2.5, label='Memory Usage (%)'),
    Line2D([0], [0], color='black', lw=2.5, linestyle='dashed', label='Predicted Forecast (%)')
]
for status, color in status_to_color.items():
    if status in df_rs['status_risk_score'].values:
        legend_elements.append(Line2D([0], [0], color=color, lw=2.5, label=f'Risk: {status}'))

ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Continuous Memory Usage and Forecast Over Time (Colored by Risk)')
plt.xlabel('Timestamp / Index')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.savefig(r'C:\AI-Predictive-Maintenance\memory\results\risk_scoring.png')
plt.show()