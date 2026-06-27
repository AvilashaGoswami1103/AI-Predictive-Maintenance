import pandas as pd
import numpy as np

def resample(df):
    if 'ts' not in df.columns:
        return df
        
    df['ts'] = pd.to_datetime(df['ts'], format='mixed', utc=True).dt.tz_localize(None)
    
    def _resample_group(group):
        group = group.sort_values('ts').set_index('ts')
        # Select only numeric columns for mean() to avoid errors with string columns
        numeric_group = group.select_dtypes(include=[np.number])
        resampled = numeric_group.resample('5min').mean()
        resampled = resampled.interpolate(method='time')
        return resampled

    resampled_df = _resample_group(df)
    resampled_df = resampled_df.reset_index(drop=False)
        
    return resampled_df

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    parser = argparse.ArgumentParser(description="Run resampling on input data and plot results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="results", help="Directory to save plot")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print("Running resampling...")
    resampled_df = resample(df)

    print("saving resampled data to csv")
    resampled_df.to_csv("resampled.csv")
        
    print("Generating plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))
    
    # Check if memory_usage_pct exists
    if 'memory_usage_pct' in df.columns and 'ts' in df.columns:
        # Pre-convert ts for raw plot
        df['ts'] = pd.to_datetime(df['ts'], format='mixed', utc=True).dt.tz_localize(None)
        
        if 'host_id' in df.columns:
            hosts = df['host_id'].unique()[:1] # plot only first host for clarity
            df_plot = df[df['host_id'] == hosts[0]]
            res_plot = resampled_df[resampled_df['host_id'] == hosts[0]]
            plt.title(f'Raw vs Resampled Memory Usage (Host {hosts[0]})')
        else:
            df_plot = df
            res_plot = resampled_df
            plt.title('Raw vs Resampled Memory Usage')
            
        plt.scatter(df_plot['ts'], df_plot['memory_usage_pct'], color='red', alpha=0.5, label='Raw Data', s=15)
        plt.plot(res_plot['ts'], res_plot['memory_usage_pct'], color='blue', alpha=0.7, label='Resampled Data')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(args.output_dir, 'resampling_comparison.png')
        plt.savefig(plot_path)
        plt.show()
        print(f"Plot saved to {plot_path}")
    else:
        print("Required columns (ts, memory_usage_pct) not found for plotting.")
