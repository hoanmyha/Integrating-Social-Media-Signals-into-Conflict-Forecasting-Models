import pandas as pd

# Input: Twitter-era dataset from Step 2
input_path = "/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_twitter_era.csv"

# Output: baseline dataset without Twitter features
output_path = "/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_twitter_era_baseline.csv"

# Load data
df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# Twitter columns to remove
twitter_cols = [
    "num_unrest_0_month",
    "num_unrest_1_month",
    "pct_unrest_1_month",
    "pct_unrest_1_month_of_year"
]

# Check that all columns exist before dropping
missing_cols = [col for col in twitter_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"These columns were not found in the dataset: {missing_cols}")

# Drop Twitter columns
df_baseline = df.drop(columns=twitter_cols)

print("Baseline shape:", df_baseline.shape)
print("Columns removed:", twitter_cols)

# Save
df_baseline.to_csv(output_path, index=False)

print("Saved baseline dataset to:", output_path)
