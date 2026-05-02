import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# =========================================================
# 1. FILE PATHS
# =========================================================
DATA_DIR = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/Input data/split_parquets")
OUT_DIR = Path("/mnt/c/Users/hahoa/OneDrive/Desktop/social_unrest_research/mini_ensemble_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_TRAIN = DATA_DIR / "baseline_train_409_460.parquet"
BASELINE_TEST  = DATA_DIR / "baseline_test_461_480.parquet"
TWITTER_TRAIN  = DATA_DIR / "twitter_train_409_460.parquet"
TWITTER_TEST   = DATA_DIR / "twitter_test_461_480.parquet"

# focus countries (from your data)
TARGET_COUNTRIES = [172, 237, 135, 138, 209, 13, 243, 233, 145]  # MDG, KEN, NPL, BGD, IDN, PER, MAR, SRB, PHL

# =========================================================
# 2. TARGET COLUMN
# =========================================================
TARGET = "lr_ged_sb"

# =========================================================
# 3. COLUMNS TO DROP FROM FEATURES
# =========================================================
DROP_COLS_COMMON = {
    TARGET,
    "year",
    "month",
    "isoab",
    "country",
}

# =========================================================
# 4. THREE SUBMODELS = MINI ENSEMBLE
# =========================================================
SUBMODELS = {
    "bittersweet_symphony": {
        "random_state": 42,
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "brown_cheese": {
        "random_state": 43,
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "car_radio": {
        "random_state": 44,
        "n_estimators": 250,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
    },
}

# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index()
    print(f"Loaded {path.name}: {df.shape}")
    return df

def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # keep only columns that exist in both train and test
    common_cols = sorted(set(train_df.columns).intersection(set(test_df.columns)))
    train_df = train_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in training data.")
    if TARGET not in test_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in test data.")

    feature_cols = []
    for col in train_df.columns:
        if col in DROP_COLS_COMMON:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            feature_cols.append(col)

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    # remove inf values and turn them into NaN
    y_train = y_train.replace([np.inf, -np.inf], np.nan)
    y_test = y_test.replace([np.inf, -np.inf], np.nan)

    # keep only rows where target is valid
    train_mask = y_train.notna()
    test_mask = y_test.notna()

    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()

    X_test = X_test.loc[test_mask].copy()
    y_test = y_test.loc[test_mask].copy()

    # clip negative values just in case
    y_train = np.clip(y_train, 0, None)
    y_test = np.clip(y_test, 0, None)

    return X_train, X_test, y_train, y_test, feature_cols

def crps_deterministic(y_true, y_pred):
    # for one point prediction, CRPS behaves like mean absolute error
    return np.mean(np.abs(y_true - y_pred))

def evaluate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None)

    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true + 1e-9, y_pred + 1e-9)
    rmsle = np.sqrt(msle)
    crps = crps_deterministic(y_true, y_pred)
    y_hat_bar = float(np.mean(y_pred))

    return {
        "RMSLE": float(rmsle),
        "CRPS": float(crps),
        "MSE": float(mse),
        "MSLE": float(msle),
        "y_hat_bar": y_hat_bar,
    }

def run_ensemble(train_path: Path, test_path: Path, label: str):
    train_df = load_df(train_path)
    test_df = load_df(test_path)

    # keep only Madagascar, Kenya, and Nepal
    train_df = train_df[train_df["country_id"].isin(TARGET_COUNTRIES)].copy()
    test_df = test_df[test_df["country_id"].isin(TARGET_COUNTRIES)].copy()

    print(f"\n[{label}] after country filter - train shape: {train_df.shape}")
    print(f"[{label}] after country filter - test shape: {test_df.shape}")

    X_train, X_test, y_train, y_test, feature_cols = prepare_xy(train_df, test_df)

    print(f"\n[{label}] number of features: {len(feature_cols)}")
    print(f"[{label}] first 15 features: {feature_cols[:15]}")

    pred_store = {}
    submodel_metrics = {}

    for model_name, params in SUBMODELS.items():
        print(f"\nTraining {label} -> {model_name}")

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            **params,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)

        pred_store[model_name] = preds
        submodel_metrics[model_name] = evaluate_metrics(y_test, preds)

    # average predictions from 3 submodels
    pred_matrix = np.column_stack([pred_store[name] for name in SUBMODELS.keys()])
    ensemble_preds = pred_matrix.mean(axis=1)

    ensemble_metrics = evaluate_metrics(y_test, ensemble_preds)

    # save prediction file
    pred_df = test_df[["month_id", "country_id"]].copy()
    pred_df["actual"] = y_test
    for model_name in SUBMODELS.keys():
        pred_df[f"pred_{model_name}"] = pred_store[model_name]
    pred_df["pred_ensemble"] = ensemble_preds

    pred_out = OUT_DIR / f"{label}_test_predictions.csv"
    pred_df.to_csv(pred_out, index=False)
    print(f"Saved predictions to: {pred_out}")

    return {
        "label": label,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "submodel_metrics": submodel_metrics,
        "ensemble_metrics": ensemble_metrics,
    }

# =========================================================
# RUN TWITTER MODEL ONLY
# =========================================================
twitter_results = run_ensemble(TWITTER_TRAIN, TWITTER_TEST, "twitter")

print("\n==============================")
print("TWITTER MODEL RESULTS")
print("==============================")
print(pd.DataFrame([twitter_results["ensemble_metrics"]]))

print("\nDone.")
