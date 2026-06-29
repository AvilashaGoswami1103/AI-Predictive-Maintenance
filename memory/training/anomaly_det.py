import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    """
    Trains and infers anomalies using Isolation Forest on specified features.
    """
    data = data.copy()
    
    # We must handle NaNs for Isolation Forest
    # Drop rows where critical features are NaN, usually resulting from rolling windows
    features = ["memory_usage_pct", "rolling_std_24h", "growth_rate", "acceleration"]
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    if data_clean.empty:
        return data
        
    X = data_clean[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    model = IsolationForest(
        contamination="auto",
        random_state=42
    )
    
    model.fit(X_scaled)
    scores = model.decision_function(X_scaled)
    
    data_clean["anomaly_score"] = -scores
    
    # Cap non-anomalies to baseline to avoid false anomaly risk contributions
    data_clean.loc[data_clean["anomaly_score"] <= 0.22, "anomaly_score"] = -0.3
    
    # Ensure major jumps are always flagged as critical anomalies
    data_clean.loc[data_clean["growth_rate"].abs() >= 0.5, "anomaly_score"] = 0.3
    
    # Merge back to original dataframe to maintain size/index
    data["anomaly_score"] = np.nan
    data.loc[data_clean.index, "anomaly_score"] = data_clean["anomaly_score"]
    
    # Save the scaler and model for inference
    import joblib
    import os
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "anomaly_det_scaler.pkl"))
    joblib.dump(model, os.path.join(model_dir, "anomaly_det_model.pkl"))
    
    return data

def plot_anomaly_detection_actual(result_df, output_dir, show_plot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import pandas as pd

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    
    if 'memory_usage_pct' in result_df.columns and 'anomaly_score' in result_df.columns and 'ts' in result_df.columns:
        result_df = result_df.copy()
        result_df['ts'] = pd.to_datetime(result_df['ts'], format='mixed', utc=True).dt.tz_localize(None)
        
        if 'host_id' in result_df.columns:
            hosts = result_df['host_id'].unique()[:1]
            df_plot = result_df[result_df['host_id'] == hosts[0]]
            plt.title(f'Actual Memory Usage Anomaly Detection (Host {hosts[0]})')
        else:
            df_plot = result_df
            plt.title('Actual Memory Usage Anomaly Detection')
            
        # Plot all points
        plt.plot(df_plot['ts'], df_plot['memory_usage_pct'], color='blue', alpha=0.5, label='Memory Usage')
        
        # Highlight anomalies (scores > 0.22)
        anomalies = df_plot[df_plot['anomaly_score'] > 0.22]
        if not anomalies.empty:
            plt.scatter(anomalies['ts'], anomalies['memory_usage_pct'], color='red', label='Anomalies', s=30, zorder=5)
            
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'anomaly_detection_actual.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    else:
        print("Required columns (ts, memory_usage_pct, anomaly_score) not found for plotting.")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run anomaly detection on actual data and plot results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="results", help="Directory to save plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print("Running anomaly detection...")
    result_df = detect_anomalies(df)
    
    print("Generating plot...")
    plot_anomaly_detection_actual(result_df, args.output_dir, show_plot=True)
