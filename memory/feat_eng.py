import pandas as pd

def engineer_features(data):
    """
    Applies feature engineering grouped by host_id.
    """
    if 'ts' in data.columns:
        data['ts'] = pd.to_datetime(data['ts'], format='mixed', utc=True).dt.tz_localize(None)
        data = data.sort_values(by='ts')
        data = data.set_index('ts')
        
        def calculate_host_features(group):
            # Ensure chronological order
            group = group.sort_index()
            group['rolling_mean_1h'] = group['memory_usage_pct'].rolling('1h').mean()
            group['rolling_mean_24h'] = group['memory_usage_pct'].rolling('24h').mean()
            group['rolling_std_1h'] = group['memory_usage_pct'].rolling('1h').std()
            group['rolling_std_24h'] = group['memory_usage_pct'].rolling('24h').std()
            
            group['growth_rate'] = group['memory_usage_pct'].pct_change()
            group['acceleration'] = group['growth_rate'].diff()
            
            group['Z_score'] = (group['memory_usage_pct'] - group['rolling_mean_24h']) / group['rolling_std_24h']
            group['trend'] = group['rolling_mean_1h'] - group['rolling_mean_24h']
            group['volatility_ratio'] = group['rolling_std_1h'] - group['rolling_std_24h']
            
            return group

        # Apply features per host to avoid cross-host leakage
        if 'host_id' in data.columns:
            data = data.groupby('host_id', group_keys=False).apply(calculate_host_features)
        else:
            # Fallback if no host_id
            data = calculate_host_features(data)

        data = data.reset_index(drop=False)
        
    return data