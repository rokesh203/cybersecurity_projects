"""
HyperMD-Enhanced - Module 4A: LSTM Model for Time-Series Performance Data
Trains an LSTM to classify sequences of system behavior as
normal (0) or suspicious (1).

Save at: D:\Rokesh Project\HyperMD-Enhanced\models\lstm_model.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASET_PATH = "processed_data/final_dataset.csv"
MODEL_OUTPUT_PATH = "models/saved_models/lstm_model.h5"

SEQUENCE_LENGTH = 10   # number of consecutive time steps per sample
FEATURE_COLUMNS = [
    "cpu_percent", "memory_percent", "process_count",
    "disk_read_bytes", "disk_write_bytes",
]
LABEL_COLUMN = "label"

EPOCHS = 20
BATCH_SIZE = 16


def load_dataset():
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run Module 3 preprocessing first."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def build_sequences(df, feature_cols, label_col, seq_length):
    """
    Convert a flat time-series table into overlapping sequences.
    Each sequence = seq_length consecutive rows of features.
    Its label = the label of the LAST row in that sequence
    (i.e., "is the system suspicious right now, given recent history?").
    """
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        print(f"WARNING: missing columns skipped: {missing}")

    data = df[available_cols].values
    labels = df[label_col].values

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(labels[i + seq_length - 1])

    X = np.array(X)
    y = np.array(y)
    print(f"Built {len(X)} sequences of length {seq_length}.")
    return X, y


def build_lstm_model(input_shape):
    """
    A simple but effective LSTM architecture:
    - LSTM layer to learn temporal patterns
    - Dropout to reduce overfitting
    - Dense layers to make the final normal/suspicious decision
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),  # outputs probability of "suspicious"
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 60)
    print("HyperMD-Enhanced - LSTM Model Training (Module 4A)")
    print("=" * 60)

    df = load_dataset()
    X, y = build_sequences(df, FEATURE_COLUMNS, LABEL_COLUMN, SEQUENCE_LENGTH)

    if len(X) == 0:
        print("ERROR: Not enough data to build sequences. Collect more data first.")
        return

    # Check class balance - important since our current labels are
    # rule-based placeholders and may be imbalanced (few suspicious rows)
    unique, counts = np.unique(y, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(unique) > 1 else None
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print("\nEvaluating model...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {os.path.abspath(MODEL_OUTPUT_PATH)}")


if __name__ == "__main__":
    main()