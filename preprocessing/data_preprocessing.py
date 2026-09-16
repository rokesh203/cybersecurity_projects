"""
HyperMD-Enhanced - Module 3: Data Preprocessing
Cleans, merges, scales, and labels collected data for machine learning.

Save at: D:\Rokesh Project\HyperMD-Enhanced\preprocessing\data_preprocessing.py
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PERFORMANCE_FILE = "performance_data.csv"
PROCESS_FILE = "process_data.csv"
SYSTEM_MEMORY_FILE = "system_memory_data.csv"

OUTPUT_FOLDER = "processed_data"
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "final_dataset.csv")
OUTPUT_JSON = os.path.join(OUTPUT_FOLDER, "final_dataset.json")

# Thresholds used for synthetic/rule-based labeling.
# These are placeholders until you have real suspicious samples
# from controlled testing - adjust based on what you observe
# in your own baseline data.
CPU_SUSPICIOUS_THRESHOLD = 80.0     # percent
MEMORY_SUSPICIOUS_THRESHOLD = 85.0  # percent
PROCESS_COUNT_SPIKE_THRESHOLD = 1.5  # multiplier over rolling average


def load_csv_safe(file_path):
    """Load a CSV file if it exists, otherwise return an empty DataFrame."""
    if not os.path.isfile(file_path):
        print(f"WARNING: {file_path} not found. Skipping.")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path}: {len(df)} rows.")
    return df


def clean_performance_data(df):
    """Clean the system performance dataset."""
    if df.empty:
        return df

    df = df.drop_duplicates()
    df = df.dropna(subset=["timestamp"])

    numeric_cols = [
        "cpu_percent", "memory_percent", "process_count",
        "disk_read_bytes", "disk_write_bytes"
    ]
    for col in numeric_cols:
        if col in df.columns:
            # Fill missing numeric values with the column median,
            # which is more robust to outliers than the mean.
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df


def clean_process_data(df):
    """Clean the process-level dataset."""
    if df.empty:
        return df

    df = df.drop_duplicates()
    df = df.dropna(subset=["timestamp", "pid"])

    numeric_cols = ["cpu_percent", "memory_mb"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

    # Replace access-denied / unknown text markers with NaN,
    # then fill with a clear placeholder for ML use.
    text_cols = ["exe_path", "username"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].replace(
                ["ACCESS_DENIED", "UNKNOWN", "", "nan"], np.nan
            )
            df[col] = df[col].fillna("UNKNOWN")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df


def clean_system_memory_data(df):
    """Clean the system memory dataset."""
    if df.empty:
        return df

    df = df.drop_duplicates()
    df = df.dropna(subset=["timestamp"])

    numeric_cols = [
        "total_mb", "available_mb", "used_mb", "free_mb",
        "percent_used", "swap_total_mb", "swap_used_mb", "swap_percent"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df


def create_labels(df):
    """
    Create a binary label column: 0 = normal, 1 = suspicious.

    NOTE: This is a rule-based placeholder labeling approach since
    no real malware samples exist yet. It flags unusually high CPU
    or memory usage as "suspicious" so you have a working dataset
    to build and test the ML pipeline structure. Once you run
    controlled test scenarios (Module 4 onward), you should replace
    this with labels based on your actual known test conditions.
    """
    if df.empty:
        return df

    df["label"] = 0

    if "cpu_percent" in df.columns:
        df.loc[df["cpu_percent"] >= CPU_SUSPICIOUS_THRESHOLD, "label"] = 1

    if "memory_percent" in df.columns:
        df.loc[df["memory_percent"] >= MEMORY_SUSPICIOUS_THRESHOLD, "label"] = 1

    if "process_count" in df.columns:
        rolling_avg = df["process_count"].rolling(window=10, min_periods=1).mean()
        spike_mask = df["process_count"] > (rolling_avg * PROCESS_COUNT_SPIKE_THRESHOLD)
        df.loc[spike_mask, "label"] = 1

    normal_count = (df["label"] == 0).sum()
    suspicious_count = (df["label"] == 1).sum()
    print(f"Labeling complete: {normal_count} normal, {suspicious_count} suspicious.")

    return df


def scale_features(df, feature_cols):
    """Scale numeric features to a 0-1 range using MinMaxScaler."""
    existing_cols = [c for c in feature_cols if c in df.columns]
    if not existing_cols:
        return df

    scaler = MinMaxScaler()
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    print(f"Scaled columns: {existing_cols}")
    return df


def merge_datasets(performance_df, system_memory_df):
    """
    Merge performance and system memory data on nearest timestamp.
    Process-level data is kept separate since it has a different
    granularity (many rows per timestamp, one per process).
    """
    if performance_df.empty:
        return performance_df

    if system_memory_df.empty:
        return performance_df

    performance_df = performance_df.sort_values("timestamp")
    system_memory_df = system_memory_df.sort_values("timestamp")

    merged = pd.merge_asof(
        performance_df,
        system_memory_df,
        on="timestamp",
        direction="nearest",
        suffixes=("", "_sysmem"),
    )

    print(f"Merged dataset: {len(merged)} rows.")
    return merged


def save_outputs(df, csv_path, json_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV -> {os.path.abspath(csv_path)}")

    # Convert timestamps to strings so JSON serialization works
    json_df = df.copy()
    for col in json_df.columns:
        if pd.api.types.is_datetime64_any_dtype(json_df[col]):
            json_df[col] = json_df[col].astype(str)

    records = json_df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved JSON -> {os.path.abspath(json_path)}")


def main():
    print("=" * 60)
    print("HyperMD-Enhanced - Data Preprocessing (Module 3)")
    print("=" * 60)

    performance_df = load_csv_safe(PERFORMANCE_FILE)
    process_df = load_csv_safe(PROCESS_FILE)
    system_memory_df = load_csv_safe(SYSTEM_MEMORY_FILE)

    print("\nCleaning data...")
    performance_df = clean_performance_data(performance_df)
    process_df = clean_process_data(process_df)
    system_memory_df = clean_system_memory_data(system_memory_df)

    print("\nMerging performance and system memory data...")
    merged_df = merge_datasets(performance_df, system_memory_df)

    print("\nCreating labels...")
    merged_df = create_labels(merged_df)

    print("\nScaling features...")
    feature_cols = [
        "cpu_percent", "memory_percent", "process_count",
        "disk_read_bytes", "disk_write_bytes",
        "percent_used", "available_mb", "used_mb",
    ]
    merged_df = scale_features(merged_df, feature_cols)

    print("\nSaving outputs...")
    save_outputs(merged_df, OUTPUT_CSV, OUTPUT_JSON)

    print("\nDone. Summary:")
    print(f"Final dataset shape: {merged_df.shape}")
    print(merged_df.head())


if __name__ == "__main__":
    main()