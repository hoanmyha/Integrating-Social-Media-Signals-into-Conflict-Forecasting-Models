import pandas as pd

# Input file from Step 1
input_path = "/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_with_twitter_filled.csv"

# Output file for Step 2
output_path = "/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_twitter_era.csv"

# Load data
df = pd.read_csv(input_path)

print("Original shape:", df.shape)
print("Original month_id range:", df["month_id"].min(), "to", df["month_id"].max())

# Filter to Twitter era only
df_twitter_era = df[(df["month_id"] >= 409) & (df["month_id"] <= 480)].copy()

print("Filtered shape:", df_twitter_era.shape)
print("Filtered month_id range:", df_twitter_era["month_id"].min(), "to", df_twitter_era["month_id"].max())

# Optional checks
twitter_cols = [
    "num_unrest_0_month",
    "num_unrest_1_month",
    "pct_unrest_1_month",
    "pct_unrest_1_month_of_year"
]

print("\nMissing values in Twitter columns:")
print(df_twitter_era[twitter_cols].isna().sum())

print("\nNon-zero counts in Twitter columns:")
print((df_twitter_era[twitter_cols] != 0).sum())

# Save
df_twitter_era.to_csv(output_path, index=False)
print("\nSaved Twitter-era dataset to:", output_path)
