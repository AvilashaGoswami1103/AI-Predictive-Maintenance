import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

def predict_usage(data, forecast_horizon=30):
    """
    Trains XGBoost models to predict memory usage and returns the predicted data.
    """
    df = data.copy()
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], format='mixed', utc=True).dt.tz_localize(None)
        df = df.sort_values('ts')
    
    df['orig_rolling_std_24h'] = df['rolling_std_24h']
    df['orig_growth_rate'] = df['growth_rate']
    df['orig_acceleration'] = df['acceleration']
    
    leakage_cols = ['rolling_mean_1h', 'rolling_mean_24h', 'rolling_std_1h', 'rolling_std_24h', 
                    'growth_rate', 'acceleration', 'Z_score', 'trend', 'volatility_ratio']
                    
    if 'host_id' in df.columns:
        df[leakage_cols] = df.groupby('host_id')[leakage_cols].shift(1)
        df['target_forecast'] = df.groupby('host_id')['memory_usage_pct'].shift(-forecast_horizon)
    else:
        df[leakage_cols] = df[leakage_cols].shift(1)
        df['target_forecast'] = df['memory_usage_pct'].shift(-forecast_horizon)
        
    df_clean = df.dropna()
    if df_clean.empty:
        return df_clean
        
    drop_cols = ['id', 'ts', 'status', 'host_id', 'target_forecast', 'orig_rolling_std_24h', 'orig_growth_rate', 'orig_acceleration', 'anomaly_score']
    features = [c for c in df_clean.columns if c not in drop_cols]
    
    X = df_clean[features]
    y = df_clean['target_forecast']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    model = xgb.XGBRegressor(
        n_estimators=2000, max_depth=8, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    
    model.fit(X_train, y_train, verbose=False)
    y_pred_test = model.predict(X_test)
    
    # Save results
    results_cols = ['ts', 'memory_usage_pct', 'orig_rolling_std_24h', 'orig_growth_rate', 'orig_acceleration']
    if 'id' in df_clean.columns: results_cols.insert(0, 'id')
    if 'host_id' in df_clean.columns: results_cols.insert(2, 'host_id')
        
    results_df = df_clean.loc[X_test.index, results_cols].copy()
    if 'host_id' not in results_df.columns:
        results_df['host_id'] = 1 # fallback
        
    results_df = results_df.rename(columns={
        'orig_rolling_std_24h': 'rolling_std_24h',
        'orig_growth_rate': 'growth_rate',
        'orig_acceleration': 'acceleration'
    })
    
    results_df['actual_forecast'] = y_test
    results_df['predicted_forecast'] = y_pred_test
    
    # Calculate rolling std, growth rate, acceleration for predicted forecast
    results_df = results_df.sort_values('ts').set_index('ts')
    
    def calc_pred_features(group):
        group['predicted_rolling_std_24h'] = group['predicted_forecast'].rolling('24h').std()
        group['predicted_growth_rate'] = group['predicted_forecast'].pct_change()
        group['predicted_acceleration'] = group['predicted_growth_rate'].diff()
        return group
        
    results_df = results_df.groupby('host_id', group_keys=False).apply(calc_pred_features)
    results_df = results_df.reset_index()
    
    return results_df

def plot_usage_prediction(result_df, output_dir, show_plot=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import pandas as pd

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    
    if 'actual_forecast' in result_df.columns and 'predicted_forecast' in result_df.columns and 'ts' in result_df.columns:
        result_df = result_df.copy()
        result_df['ts'] = pd.to_datetime(result_df['ts'], format='mixed', utc=True).dt.tz_localize(None)
        
        if 'host_id' in result_df.columns:
            hosts = result_df['host_id'].unique()[:1]
            df_plot = result_df[result_df['host_id'] == hosts[0]]
            plt.title(f'Actual vs Predicted Usage (Test Set) (Host {hosts[0]})')
        else:
            df_plot = result_df
            plt.title('Actual vs Predicted Usage (Test Set)')
            
        plt.plot(df_plot['ts'], df_plot['actual_forecast'], color='blue', alpha=0.7, label='Actual Future Usage')
        plt.plot(df_plot['ts'], df_plot['predicted_forecast'], color='red', alpha=0.7, linestyle='--', label='Predicted Future Usage')
        
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.legend()
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'usage_prediction.png')
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    else:
        print("Required columns (ts, actual_forecast, predicted_forecast) not found for plotting.")

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run usage prediction on input data and plot results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="results", help="Directory to save plot")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print(f"Running usage prediction (horizon={args.horizon})...")
    result_df = predict_usage(df, forecast_horizon=args.horizon)
    
    print("Generating plot...")
    plot_usage_prediction(result_df, args.output_dir, show_plot=True)