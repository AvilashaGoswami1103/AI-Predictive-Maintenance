import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def predict_anomalies(data):
    """
    Detects anomalies on predicted forecasts using Isolation Forest.
    """
    data = data.copy()
    
    # Drop rows where critical predicted features are NaN
    features = ["predicted_forecast", "predicted_rolling_std_24h", "predicted_growth_rate", "predicted_acceleration"]
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
    
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def predict_anomalies(data):
    """
    Detects anomalies on predicted forecasts using Isolation Forest.
    """
    data = data.copy()
    
    # Drop rows where critical predicted features are NaN
    features = ["predicted_forecast", "predicted_rolling_std_24h", "predicted_growth_rate", "predicted_acceleration"]
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
    
    # Merge back to original data
    data["anomaly_score"] = np.nan
    data.loc[data_clean.index, "anomaly_score"] = data_clean["anomaly_score"]
    
    return data

def plot_anomaly_prediction(result_df, output_dir, show_plot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import pandas as pd

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    
    if 'predicted_forecast' in result_df.columns and 'anomaly_score' in result_df.columns and 'ts' in result_df.columns:
        result_df = result_df.copy()
        result_df['ts'] = pd.to_datetime(result_df['ts'], format='mixed', utc=True).dt.tz_localize(None)
        
        if 'host_id' in result_df.columns:
            hosts = result_df['host_id'].unique()[:1]
            df_plot = result_df[result_df['host_id'] == hosts[0]]
            plt.title(f'Predicted Memory Usage Anomaly Detection (Host {hosts[0]})')
        else:
            df_plot = result_df
            plt.title('Predicted Memory Usage Anomaly Detection')
            
        plt.plot(df_plot['ts'], df_plot['predicted_forecast'], color='blue', alpha=0.5, linestyle='--', label='Predicted Forecast')
        
        anomalies = df_plot[df_plot['anomaly_score'] > 0.22]
        if not anomalies.empty:
            plt.scatter(anomalies['ts'], anomalies['predicted_forecast'], color='red', label='Predicted Anomalies', s=30, zorder=5)
            
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'anomaly_prediction.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    else:
        print("Required columns (ts, predicted_forecast, anomaly_score) not found for plotting.")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run anomaly detection on predicted data and plot results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="results", help="Directory to save plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print("Running anomaly prediction...")
    result_df = predict_anomalies(df)
    
    print("Generating plot...")
    plot_anomaly_prediction(result_df, args.output_dir, show_plot=True)
