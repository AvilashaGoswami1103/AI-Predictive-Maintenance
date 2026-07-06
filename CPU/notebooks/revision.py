import pandas as pd

# ==========================
# Load Dataset
# ==========================
input_file = r"C:\Users\Avilasha\Desktop\CPU_Predictive_Maintenance\CPU\data\Processed\cpu_baseline.csv"      # Change to your input filename
output_file = r"C:\Users\Avilasha\Desktop\CPU_Predictive_Maintenance\CPU\data\Features\cpu_baseline_revised.csv"

df = pd.read_csv(input_file)

# Saturday should NOT be weekend
df.loc[df["day_of_week"] == 5, "is_weekend"] = 0

# ==========================
# Convert timestamp column to datetime
# ==========================
# Replace "timestamp" with "ts" if needed
if "timestamp" in df.columns:
    datetime_col = "timestamp"
elif "ts" in df.columns:
    datetime_col = "ts"
else:
    raise ValueError("No timestamp column found.")

df[datetime_col] = pd.to_datetime(
    df[datetime_col],
    format="mixed",
    utc=True
)

# ==========================
# Create working_hour column
# ==========================
weekday = df[datetime_col].dt.weekday  # Monday=0 ... Sunday=6

df["working_hour"] = 0

# Monday to Friday: 9 AM - 6 PM
weekday_mask = (
    (weekday <= 4) &
    (df["hour_of_day"] >= 9) &
    (df["hour_of_day"] < 18)
)

# Saturday: 9 AM - 1 PM
saturday_mask = (
    (weekday == 5) &
    (df["hour_of_day"] >= 9) &
    (df["hour_of_day"] < 13)
)

df.loc[weekday_mask | saturday_mask, "working_hour"] = 1

# ==========================
# Remove day_of_week column
# ==========================
if "day_of_week" in df.columns:
    df = df.drop(columns=["day_of_week"])

# ==========================
# Save as a NEW CSV
# ==========================
df.to_csv(output_file, index=False)

print(f"New processed file saved as: {output_file}")
print(df.head())