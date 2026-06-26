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

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    parser = argparse.ArgumentParser(description="Run feature engineering on input data and plot results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="results", help="Directory to save plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print("Running feature engineering...")
    result_df = engineer_features(df)

    print("saving the feature engineered data")
    result_df.to_csv("feat_eng.csv")
    
    print("Generating plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    
    if 'memory_usage_pct' in result_df.columns and 'rolling_mean_24h' in result_df.columns and 'ts' in result_df.columns:
        result_df['ts'] = pd.to_datetime(result_df['ts'], format='mixed', utc=True).dt.tz_localize(None)
        
        if 'host_id' in result_df.columns:
            hosts = result_df['host_id'].unique()[:1]
            df_plot = result_df[result_df['host_id'] == hosts[0]].copy()
            plt.title(f'Memory Usage vs 24h Rolling Mean (Host {hosts[0]})')
        else:
            df_plot = result_df.copy()
            plt.title('Memory Usage vs 24h Rolling Mean')
            
        plt.plot(df_plot['ts'], df_plot['memory_usage_pct'], color='blue', alpha=0.5, label='Memory Usage (%)')
        plt.plot(df_plot['ts'], df_plot['rolling_mean_24h'], color='red', alpha=0.8, linewidth=2, label='24h Rolling Mean')
        
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(args.output_dir, 'feature_engineering.png')
        plt.savefig(plot_path)
        plt.show()
        print(f"Plot saved to {plot_path}")
    else:
        print("Required columns (ts, memory_usage_pct, rolling_mean_24h) not found for plotting.")