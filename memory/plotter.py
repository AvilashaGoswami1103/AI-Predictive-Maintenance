import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


data = pd.read_excel(r"C:\AI-Predictive-Maintenance\memory\results.xlsx")

data['ts'] = pd.to_datetime(data['ts'], format='mixed')


# Filter for anomalies
anomalies = data[data['anomaly_score'] >= 0.21]

fig = go.Figure()

# 1. Plot the full memory usage timeline as a line
fig.add_trace(
    go.Scatter(x=data['ts'], y=data['memory_usage_pct'], 
               mode='lines', name='Memory Usage', line=dict(color='steelblue', width=1.5))
)

# 2. Add the anomaly points where score >= 0.22
fig.add_trace(
    go.Scatter(x=anomalies['ts'], y=anomalies['memory_usage_pct'],
               mode='markers', name='Anomaly (score >= 0.22)',
               marker=dict(color='red', size=8, line=dict(color='black', width=1)))
)

fig.update_layout(title='Memory Usage over Time with Anomalies',
                  xaxis_title='Time',
                  yaxis_title='Memory Usage (%)')

fig.show()
