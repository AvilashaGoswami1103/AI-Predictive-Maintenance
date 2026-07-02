import sys
import os
import argparse
import pandas as pd

# Add the training directory to sys.path to reuse its stateless modules
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_dir = os.path.join(base_dir, "training")
sys.path.append(training_dir)

from resampling import resample
from feat_eng import engineer_features
from risk_scoring import calculate_risk
from anomaly_det import plot_anomaly_detection_actual
from usage_pred import plot_usage_prediction
from anomaly_pred import plot_anomaly_prediction
from compare import generate_comparison_plots

from inference_anomaly_det import detect_anomalies_inference
from inference_usage_pred import predict_usage_inference
from inference_anomaly_pred import predict_anomalies_inference
from deviation_det import drift_detection

def run_inference_pipeline(input_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
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
    data.drop(columns=["cpu_usage_pct", "status", "power_kw"], inplace=True, errors='ignore')
    
    print("Running resampling...")
    data = resample(data)
    if data.empty:
        print("Data is empty after resampling. Exiting.")
        return
        
    print("Running feature engineering...")
    data = engineer_features(data)
    if data.empty:
        print("Data is empty after feature engineering. Exiting.")
        return
        
    print("Running anomaly detection inference on actual data...")
    data = detect_anomalies_inference(data)
    plot_anomaly_detection_actual(data, output_dir)
    
    print("Calculating risk scores on actual data...")
    data = calculate_risk(data, is_predicted=False)
    
    actuals_path = os.path.join(output_dir, 'inference_actual_risk_scoring.csv')
    data.to_csv(actuals_path, index=False)
    print(f"Actuals saved to {actuals_path}")
    
    print(f"Running usage prediction inference...")
    predicted_data = predict_usage_inference(data.drop(columns=['status_risk_score'], errors='ignore'))
    if predicted_data.empty:
        print("Prediction failed or returned empty. Exiting.")
        return
        
    plot_usage_prediction(predicted_data, output_dir)
        
    print("Running anomaly detection inference on predicted data...")
    predicted_data = predict_anomalies_inference(predicted_data)
    plot_anomaly_prediction(predicted_data, output_dir)
    
    print("Calculating risk scores on predicted data...")
    predicted_data = calculate_risk(predicted_data, is_predicted=True)
    
    preds_path = os.path.join(output_dir, 'inference_predicted_risk_scoring.csv')
    predicted_data.to_csv(preds_path, index=False)
    print(f"Predictions saved to {preds_path}")
    
    print("Generating comparison plots...")
    generate_comparison_plots(data, predicted_data, output_dir)
    
    print("Running drift detection...")
    drift_detection(data, predicted_data, output_dir)
    
    print("\nInference Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory Predictive Maintenance Inference Pipeline")
    parser.add_argument("--inputs", nargs="+", required=True, help="Path(s) to input CSV file(s) containing host metrics.")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Directory to save output files and plots.")
    
    args = parser.parse_args()
    
    run_inference_pipeline(args.inputs, args.output_dir)
