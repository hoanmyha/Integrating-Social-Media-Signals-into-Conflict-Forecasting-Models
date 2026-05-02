import pandas as pd
from pathlib import Path

# ----------------------------
# Input files
# ----------------------------
twitter_csv = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_twitter_era.csv")
baseline_csv = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/all_cm_features_fatalities003_121-554_twitter_era_baseline.csv")

# ----------------------------
# Output folder
# ----------------------------
outdir = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/split_parquets")
outdir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Train/test split
# ----------------------------
TRAIN_START = 409
TRAIN_END = 460
TEST_START = 461
TEST_END = 480

def prep_df(df):
    # sort rows the way ViEWS-style data expects
    df = df.sort_values(["month_id", "country_id"]).copy()

    # make month_id and country_id the index
    df = df.set_index(["month_id", "country_id"])

    return df

def split_and_save(csv_path, prefix):
    df = pd.read_csv(csv_path)

    print(f"\n=== {prefix.upper()} ===")
    print("Original shape:", df.shape)
    print("Month range:", df["month_id"].min(), "to", df["month_id"].max())

    train_df = df[(df["month_id"] >= TRAIN_START) & (df["month_id"] <= TRAIN_END)].copy()
    test_df  = df[(df["month_id"] >= TEST_START) & (df["month_id"] <= TEST_END)].copy()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("Train month range:", train_df["month_id"].min(), "to", train_df["month_id"].max())
    print("Test month range:", test_df["month_id"].min(), "to", test_df["month_id"].max())

    train_df = prep_df(train_df)
    test_df = prep_df(test_df)

    train_out = outdir / f"{prefix}_train_409_460.parquet"
    test_out = outdir / f"{prefix}_test_461_480.parquet"

    train_df.to_parquet(train_out)
    test_df.to_parquet(test_out)

    print("Saved train to:", train_out)
    print("Saved test to:", test_out)

# ----------------------------
# Run both datasets
# ----------------------------
split_and_save(baseline_csv, "baseline")
split_and_save(twitter_csv, "twitter")

print("\nDone.")
