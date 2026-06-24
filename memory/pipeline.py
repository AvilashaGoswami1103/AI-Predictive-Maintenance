import os
import argparse
import pandas as pd

from feat_eng import engineer_features
from anomaly_det import detect_anomalies
from usage_pred import predict_usage
from anomaly_pred import predict_anomalies
from risk_scoring import calculate_risk
from compare import generate_comparison_plots

def run_pipeline(input_files, output_dir, forecast_horizon=30):
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print(f"Loading data from {len(input_files)} file(s)...")
    dfs = []
    for f in input_files:
        try:
            d = pd.read_csv(f)
            dfs.append(d)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        print("No valid data loaded. Exiting.")
        return
        
    data = pd.concat(dfs, ignore_index=True)
    
    # Drop exploratory columns if present (to standardize)
    data.drop(columns=["cpu_usage_pct", "status", "power_kw"], inplace=True, errors='ignore')
    
    # ---------------------------------------------------------
    # 2. Feature Engineering
    # ---------------------------------------------------------
    print("Running feature engineering...")
    data = engineer_features(data)
    if data.empty:
        print("Data is empty after feature engineering. Exiting.")
        return
        
    # ---------------------------------------------------------
    # 3. Anomaly Detection (Actuals)
    # ---------------------------------------------------------
    print("Running anomaly detection on actual data...")
    data = detect_anomalies(data)
    
    # ---------------------------------------------------------
    # 4. Risk Scoring (Actuals)
    # ---------------------------------------------------------
    print("Calculating risk scores on actual data...")
    data = calculate_risk(data, is_predicted=False)
    
    actuals_path = os.path.join(output_dir, 'actual_risk_scoring.csv')
    data.to_csv(actuals_path, index=False)
    print(f"Actuals saved to {actuals_path}")
    
    # ---------------------------------------------------------
    # 5. Usage Prediction
    # ---------------------------------------------------------
    print(f"Running usage prediction (horizon={forecast_horizon})...")
    predicted_data = predict_usage(data.drop(columns=['status_risk_score']), forecast_horizon=forecast_horizon)
    if predicted_data.empty:
        print("Prediction failed or returned empty. Exiting.")
        return
        
    # ---------------------------------------------------------
    # 6. Anomaly Detection (Predicted)
    # ---------------------------------------------------------
    print("Running anomaly detection on predicted data...")
    predicted_data = predict_anomalies(predicted_data)
    
    # ---------------------------------------------------------
    # 7. Risk Scoring (Predicted)
    # ---------------------------------------------------------
    print("Calculating risk scores on predicted data...")
    predicted_data = calculate_risk(predicted_data, is_predicted=True)
    
    preds_path = os.path.join(output_dir, 'predicted_risk_scoring.csv')
    predicted_data.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")
    
    # ---------------------------------------------------------
    # 8. Generate Plots
    # ---------------------------------------------------------
    print("Generating comparison plots...")
    generate_comparison_plots(data, predicted_data, output_dir)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory Predictive Maintenance Pipeline")
    parser.add_argument("--inputs", nargs="+", help="Path(s) to input CSV file(s) containing host metrics.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output files and plots.")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon (number of steps).")
    
    args = parser.parse_args()
    
    if not args.inputs:
        # Fallback to local files if no arguments provided (for backward compatibility during testing)
        import glob
        default_files = glob.glob(r"C:\server_data\Datacenter Datas\SERVER WISE DATA\host_metrics_host*.csv")
        if not default_files:
            print("No inputs provided and no default files found. Use --inputs <file_paths>")
        else:
            print("Using default files found in C:\server_data\Datacenter Datas\SERVER WISE DATA\\")
            run_pipeline(default_files, args.output_dir, args.horizon)
    else:
        run_pipeline(args.inputs, args.output_dir, args.horizon)
