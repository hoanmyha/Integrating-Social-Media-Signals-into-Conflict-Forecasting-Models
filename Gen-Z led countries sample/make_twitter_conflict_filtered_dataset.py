import pandas as pd
from pathlib import Path

input_path = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_with_twitter_filled.csv")
output_path = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_with_twitter_conflict_filtered.csv")

df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# Keep rows where Twitter exists AND conflict predictor is not zero
filtered = df[
    (
        (df["num_unrest_0_month"] > 0) |
        (df["num_unrest_1_month"] > 0)
    ) &
    (df["lr_ged_sb"] > 0)
].copy()

print("Filtered shape:", filtered.shape)

print("\nNon-zero counts:")
for col in [
    "lr_ged_sb",
    "num_unrest_0_month",
    "num_unrest_1_month",
    "pct_unrest_1_month",
    "pct_unrest_1_month_of_year"
]:
    print(col, (filtered[col] != 0).sum())

filtered.to_csv(output_path, index=False)
print("\nSaved filtered file to:", output_path)
