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
            0.47 * df['predicted_instant_change'].apply(transition_risk) +
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
            0.20 * df['memory_usage_pct'].apply(forecast_risk) +
            0.40 * df['anomaly_score'].apply(anomaly_risk) +
            0.45 * df['instant_change'].apply(transition_risk) +
            0.20 * df['rolling_std_24h'].apply(stability_risk)
        )
        
    df['status_risk_score'] = df['risk_score'].apply(get_risk_status)
    return df

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    parser = argparse.ArgumentParser(description="Run risk scoring on input data and plot results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="results", help="Directory to save plot")
    parser.add_argument("--is_predicted", action="store_true", help="Flag if the data is predicted data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print("Calculating risk scores...")
    result_df = calculate_risk(df, is_predicted=args.is_predicted)
    
    print("Generating plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    
    if 'risk_score' in result_df.columns and 'ts' in result_df.columns:
        result_df['ts'] = pd.to_datetime(result_df['ts'], format='mixed', utc=True).dt.tz_localize(None)
        
        if 'host_id' in result_df.columns:
            hosts = result_df['host_id'].unique()[:1]
            df_plot = result_df[result_df['host_id'] == hosts[0]].copy()
            plt.title(f'Risk Score Over Time (Host {hosts[0]})')
        else:
            df_plot = result_df.copy()
            plt.title('Risk Score Over Time')
            
        plt.plot(df_plot['ts'], df_plot['risk_score'], color='black', alpha=0.3, label='Risk Score Line')
        
        risk_colors = {
            'healthy': 'green',
            'low risk': 'gold',
            'med risk': 'orange',
            'high risk': 'red',
            'critical': 'darkred'
        }
        
        if 'status_risk_score' in df_plot.columns:
            for status, color in risk_colors.items():
                subset = df_plot[df_plot['status_risk_score'] == status]
                if not subset.empty:
                    plt.scatter(subset['ts'], subset['risk_score'], color=color, label=f'Status: {status}', s=30, zorder=5)
                    
        plt.xlabel('Time')
        plt.ylabel('Risk Score')
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(args.output_dir, 'risk_score.png')
        plt.savefig(plot_path)
        plt.show()
        print(f"Plot saved to {plot_path}")
    else:
        print("Required columns (ts, risk_score) not found for plotting.")
