"""
===========================================================
Project : CPU Predictive Maintenance
Step    : Saturday Working Hours Processing
Author  : Avilasha
===========================================================

Working Hours Logic

Monday-Friday
    Working Hours : 09:00 - 18:00

Saturday
    is_weekend = 0
    Working Hours : 09:00 - 13:00

Sunday
    is_weekend = 1
    Working Hours : None

Output:
    cpu_baseline_processed.csv
"""

import pandas as pd

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

INPUT_FILE = r"C:\Users\Avilasha\Desktop\CPU_Predictive_Maintenance\CPU\data\Features\cpu_baseline_revised.csv"
OUTPUT_FILE = r"C:\Users\Avilasha\Desktop\CPU_Predictive_Maintenance\CPU\data\Features\cpu_baseline_revised.csv"

df = pd.read_csv(INPUT_FILE)

print("Dataset Loaded")
print(df.shape)

# ---------------------------------------------------------
# Detect Timestamp Column
# ---------------------------------------------------------

possible_timestamp_columns = [
    "ts",
    "timestamp",
    "datetime",
    "date_time"
]

timestamp_col = None

for col in possible_timestamp_columns:
    if col in df.columns:
        timestamp_col = col
        break

if timestamp_col is None:
    raise Exception("Timestamp column not found.")

# ---------------------------------------------------------
# Convert Timestamp
# ---------------------------------------------------------

df[timestamp_col] = pd.to_datetime(
    df[timestamp_col],
    format="mixed",
    utc=True
)

# ---------------------------------------------------------
# Create Time Features
# ---------------------------------------------------------

df["hour"] = df[timestamp_col].dt.hour

df["minute"] = df[timestamp_col].dt.minute

df["day_of_week"] = df[timestamp_col].dt.dayofweek
# Monday = 0
# Tuesday = 1
# Wednesday = 2
# Thursday = 3
# Friday = 4
# Saturday = 5
# Sunday = 6

# ---------------------------------------------------------
# Default Weekend
# ---------------------------------------------------------

df["is_weekend"] = 0

df.loc[df["day_of_week"] == 6, "is_weekend"] = 1

# ---------------------------------------------------------
# Working Hour Initialization
# ---------------------------------------------------------

df["working_hour"] = 0

# ---------------------------------------------------------
# Monday-Friday
# Working Hours
# 09:00 to 17:00
# ---------------------------------------------------------

weekday_mask = (
    (df["day_of_week"] <= 4)
    &
    (df["hour"] >= 9)
    &
    (df["hour"] < 18)
)

df.loc[weekday_mask, "working_hour"] = 1

# ---------------------------------------------------------
# Saturday
# Working Hours
# 09:00 to 13:00
# Saturday is NOT weekend
# ---------------------------------------------------------

df.loc[df["day_of_week"] == 5, "is_weekend"] = 0

saturday_mask = (
    (df["day_of_week"] == 5)
    &
    (df["hour"] >= 9)
    &
    (df["hour"] < 13)
)

df.loc[saturday_mask, "working_hour"] = 1

# ---------------------------------------------------------
# Sunday
# ---------------------------------------------------------

df.loc[df["day_of_week"] == 6, "working_hour"] = 0

# ---------------------------------------------------------
# Sort
# ---------------------------------------------------------

df = df.sort_values(timestamp_col)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessing Complete")

print("\nWorking Hour Distribution")

print(df["working_hour"].value_counts())

print("\nWeekend Distribution")

print(df["is_weekend"].value_counts())

print("\nSaved as")

print(OUTPUT_FILE)