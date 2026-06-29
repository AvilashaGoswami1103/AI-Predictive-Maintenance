import pandas as pd
import xgboost as xgb
import os

def predict_usage_inference(data):
    """
    Infers future memory usage using the pre-trained XGBoost model.
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
    else:
        df[leakage_cols] = df[leakage_cols].shift(1)
        
    # Drop rows where shifted features are NaN (e.g., the very first row)
    # But unlike training, we DO NOT require target_forecast to be non-NaN, so we can predict the latest rows!
    df_clean = df.dropna(subset=leakage_cols + ['memory_usage_pct'])
    if df_clean.empty:
        print("Data empty after dropping NaNs in usage prediction features.")
        return df_clean
        
    drop_cols = ['id', 'ts', 'status', 'host_id', 'target_forecast', 'orig_rolling_std_24h', 'orig_growth_rate', 'orig_acceleration', 'anomaly_score', 'status_risk_score']
    features = [c for c in df_clean.columns if c not in drop_cols]
    
    X = df_clean[features]
    
    # Load model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(model_dir, "usage_pred_model.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Usage prediction model not found at {model_path}. Please run the training pipeline first.")
        
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    y_pred = model.predict(X)
    
    # Build results
    results_cols = ['ts', 'memory_usage_pct', 'orig_rolling_std_24h', 'orig_growth_rate', 'orig_acceleration']
    if 'id' in df_clean.columns: results_cols.insert(0, 'id')
    if 'host_id' in df_clean.columns: results_cols.insert(2, 'host_id')
        
    results_df = df_clean[results_cols].copy()
    if 'host_id' not in results_df.columns:
        results_df['host_id'] = 1 # fallback
        
    results_df = results_df.rename(columns={
        'orig_rolling_std_24h': 'rolling_std_24h',
        'orig_growth_rate': 'growth_rate',
        'orig_acceleration': 'acceleration'
    })
    
    results_df['predicted_forecast'] = y_pred
    
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
