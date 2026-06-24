import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_rs = pd.read_csv(r"C:\AI-Predictive-Maintenance\memory\results\anomaly_result.csv")

# Drop any NaNs or infinite values inherited from previous steps to prevent risk functions from crashing
df_rs = df_rs.replace([np.inf, -np.inf], np.nan).dropna()
df_rs['instant_change'] = abs(df_rs['memory_usage_pct'] - df_rs['memory_usage_pct'].shift(1))

def transition_risk(instant_change):
    return min(100, instant_change * 2)


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
    0.20 * df_rs['memory_usage_pct'].apply(current_risk) +
    0.40 * df_rs['anomaly_score'].apply(anomaly_risk) +
    0.40 * df_rs['instant_change'].apply(transition_risk) +
    0.20 * df_rs['rolling_std_24h'].apply(stability_risk)
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

ax.add_collection(lc_mem)
ax.autoscale()

# Configure X-axis ticks
if x_col == 'ts':
    step = max(1, len(df_rs) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(df_rs['ts'].iloc[::step], rotation=45)

# Create a custom legend
legend_elements = [
    Line2D([0], [0], color='black', lw=2.5, label='Memory Usage (%)')
]
for status, color in status_to_color.items():
    if status in df_rs['status_risk_score'].values:
        legend_elements.append(Line2D([0], [0], color=color, lw=2.5, label=f'Risk: {status}'))

ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Continuous Memory Usage Over Time (Colored by Risk)')
plt.xlabel('Timestamp / Index')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.savefig(r'C:\AI-Predictive-Maintenance\memory\results\risk_scoring.png')
plt.show()