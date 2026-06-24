import pandas as pd
import numpy as np

def calculate_risk(data, is_predicted=False):
    """
    Calculates risk scores based on memory usage, anomaly scores, and rolling stats.
    Supports both actual data and predicted data via the is_predicted flag.
    """
    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Shared risk functions
    def transition_risk(instant_change):
        return min(100, instant_change * 2)

    def forecast_risk(memory_usage):
        if memory_usage < 50: return 10
        elif memory_usage < 70: return 30
        elif memory_usage < 86: return 70
        else: return 100

    def anomaly_risk(score, min_score=-0.3, max_score=0.3):
        risk = 100 * ((score - min_score) / (max_score - min_score))
        return np.clip(risk, 0, 100)

    def get_risk_status(score):
        if pd.isna(score): return 'unknown'
        if score < 30: return 'healthy'
        elif score < 60: return 'low risk'
        elif score < 76: return 'med risk'
        else: return 'critical'

    if is_predicted:
        if 'predicted_forecast' not in df.columns:
            return df
            
        # Group by host_id to calculate instant change properly
        if 'host_id' in df.columns:
            df['predicted_instant_change'] = df.groupby('host_id')['predicted_forecast'].diff().abs()
        else:
            df['predicted_instant_change'] = df['predicted_forecast'].diff().abs()
            
        max_std = df['predicted_rolling_std_24h'].max()
        
        def stability_risk(std):
            if pd.isna(std) or pd.isna(max_std) or max_std == 0: return 0
            return min(100, (std / max_std) * 100)
            
        df['risk_score'] = (
            0.20 * df['predicted_forecast'].apply(forecast_risk) +
            0.40 * df['anomaly_score'].apply(anomaly_risk) +
            0.45 * df['predicted_instant_change'].apply(transition_risk) +
            0.20 * df['predicted_rolling_std_24h'].apply(stability_risk)
        )
    else:
        if 'memory_usage_pct' not in df.columns:
            return df
            
        # Group by host_id to calculate instant change properly
        if 'host_id' in df.columns:
            df['instant_change'] = df.groupby('host_id')['memory_usage_pct'].diff().abs()
        else:
            df['instant_change'] = df['memory_usage_pct'].diff().abs()
            
        max_std = df['rolling_std_24h'].max()
        
        def stability_risk(std):
            if pd.isna(std) or pd.isna(max_std) or max_std == 0: return 0
            return min(100, (std / max_std) * 100)
            
        df['risk_score'] = (
            0.25 * df['memory_usage_pct'].apply(forecast_risk) +
            0.40 * df['anomaly_score'].apply(anomaly_risk) +
            0.25 * df['instant_change'].apply(transition_risk) +
            0.20 * df['rolling_std_24h'].apply(stability_risk)
        )
        
    df['status_risk_score'] = df['risk_score'].apply(get_risk_status)
    return df