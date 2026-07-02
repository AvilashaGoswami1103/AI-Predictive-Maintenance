import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os

DEVIATION_TH = 20

def drift_detection(df_real, df_predicted, output_dir=None):
    if 'ts' not in df_real.columns or 'ts' not in df_predicted.columns:
        print("Missing timestamp 'ts' column in data.")
        return None
        
    merged = pd.merge(df_real[['ts', 'memory_usage_pct']], 
                      df_predicted[['ts', 'predicted_forecast']], 
                      on='ts', how='inner')
                      
    if merged.empty:
        print("No matching timestamps found to compute drift.")
        return None
        
    merged['drift'] = np.abs(merged['memory_usage_pct'] - merged['predicted_forecast'])
    merged['drift_detected'] = merged['drift'] > DEVIATION_TH
    
    drift_points = merged[merged['drift_detected']]
    
    if not drift_points.empty:
        print(f"Drift detected at {len(drift_points)} points (Threshold: {DEVIATION_TH} units)!")
    else:
        print("No drift detected.")
        
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(merged['ts'], merged['drift'], label='Drift (Absolute Error)', color='blue')
        plt.axhline(y=DEVIATION_TH, color='red', linestyle='--', label=f'Threshold ({DEVIATION_TH})')
        plt.scatter(drift_points['ts'], drift_points['drift'], color='red', label='Drift Detected', zorder=5)
        plt.title('Memory Usage Drift Detection')
        plt.xlabel('Time')
        plt.ylabel('Drift (Units)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drift_detection.png'))
        plt.close()
        
    return merged