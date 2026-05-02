import pandas as pd
import pycountry
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------
original_path = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554.csv")
twitter_path = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/unrest_twitter_data_aggregated_by_month.csv")

output_path = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_with_twitter_filled.csv")

# -----------------------------
# Load files
# -----------------------------
orig = pd.read_csv(original_path)
tw = pd.read_csv(twitter_path)

print("Original shape:", orig.shape)
print("Twitter shape:", tw.shape)

# -----------------------------
# Keep only needed Twitter columns
# -----------------------------
tw = tw[
    [
        "year",
        "month",
        "country_code",
        "num_unrest_0_month",
        "num_unrest_1_month",
        "pct_unrest_1_month",
        "pct_unrest_1_month_of_year",
    ]
].copy()

# -----------------------------
# Convert 2-letter country code to 3-letter code
# -----------------------------
def alpha2_to_alpha3(code):
    try:
        country = pycountry.countries.get(alpha_2=str(code).upper())
        return country.alpha_3
    except:
        return None

tw["isoab"] = tw["country_code"].apply(alpha2_to_alpha3)

# -----------------------------
# Check whether any codes failed to convert
# -----------------------------
bad_codes = tw[tw["isoab"].isna()]["country_code"].drop_duplicates().tolist()
print("Unmatched Twitter country codes:", bad_codes)

# Drop rows where conversion failed
tw = tw.dropna(subset=["isoab"]).copy()

# -----------------------------
# Merge
# -----------------------------
merged = orig.merge(
    tw[
        [
            "year",
            "month",
            "isoab",
            "num_unrest_0_month",
            "num_unrest_1_month",
            "pct_unrest_1_month",
            "pct_unrest_1_month_of_year",
        ]
    ],
    on=["year", "month", "isoab"],
    how="left"
)

print("Merged shape:", merged.shape)

# -----------------------------
# Show how many rows got matches
# -----------------------------
match_count = merged["num_unrest_1_month"].notna().sum()
print("Rows with Twitter match:", match_count)
print("Rows without Twitter match:", len(merged) - match_count)

# -----------------------------
# Fill missing Twitter values with 0
# -----------------------------
twitter_cols = [
    "num_unrest_0_month",
    "num_unrest_1_month",
    "pct_unrest_1_month",
    "pct_unrest_1_month_of_year",
]

for col in twitter_cols:
    merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged[col] = merged[col].fillna(0)

print("\nMissing values after fill:")
print(merged[twitter_cols].isna().sum())

# -----------------------------
# Save merged file
# -----------------------------
merged.to_csv(output_path, index=False)
print("Saved merged file to:", output_path)
