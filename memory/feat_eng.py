import pandas as pd 

def featured_data(file_path, output_path=None):
    data = pd.read_csv(file_path)
    
    if 'ts' in data.columns:
        data['ts_dt'] = pd.to_datetime(data['ts'], format='mixed', utc=True).dt.tz_localize(None)
        data = data.sort_values(by='ts_dt')
        
        data = data.set_index('ts_dt')
        
        data['rolling_mean_1h'] = data['memory_usage_pct'].rolling('1h').mean()
        data['rolling_mean_24h'] = data['memory_usage_pct'].rolling('24h').mean()
        data['rolling_std_1h'] = data['memory_usage_pct'].rolling('1h').std()
        data['rolling_std_24h'] = data['memory_usage_pct'].rolling('24h').std()
        
        data['growth_rate'] = data['memory_usage_pct'].pct_change()
        data['acceleration'] = data['growth_rate'].diff()
        
        data['Z_score'] = (data['memory_usage_pct'] - data['rolling_mean_24h']) / data['rolling_std_24h']
        data['trend'] = data['rolling_mean_1h'] - data['rolling_mean_24h']
        data['volatility_ratio'] = data['rolling_std_1h'] - data['rolling_std_24h']
        
        data = data.reset_index(drop=True)
        
    if output_path:
        data.to_csv(output_path, index=False)
        
    return data


file1= r"C:\server_data\Datacenter Datas\SERVER WISE DATA\host_metrics_host1.csv"
file2 = r"C:\server_data\Datacenter Datas\SERVER WISE DATA\host_metrics_host2.csv"
file3 = r"C:\server_data\Datacenter Datas\SERVER WISE DATA\host_metrics_host3.csv"
featured_data(file1, "out.csv")
featured_data(file2, "out1.csv")
featured_data(file3, "out2.csv")