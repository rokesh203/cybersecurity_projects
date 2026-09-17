"""
HyperMD-Enhanced - Module 4D: Unified Inference Engine
=======================================================
Wraps the trained fusion model into a single, clean prediction
function usable by the dashboard, explainability module, and
response engine.

Given:
  - A NumPy array of recent performance readings (shape: timesteps x 5)
  - A path to a memory dump image (PNG, 256x256 grayscale)

Returns a dict:
  {
    "label"         : 0 or 1,
    "label_name"    : "normal" or "suspicious",
    "confidence"    : float in [0.0, 1.0],
    "lstm_features" : np.ndarray shape (16,),
    "cnn_features"  : np.ndarray shape (128,),
    "fused_vector"  : np.ndarray shape (144,),
  }

The feature vectors are passed to the explainability module (Step 3)
so it can compute SHAP values and Grad-CAM without re-running inference.

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\models\\inference.py

Usage (standalone)
------------------
    # Quick self-test using the first real session data
    .venv\\Scripts\\python.exe models/inference.py --selftest

Usage (import)
--------------
    from models.inference import predict, load_models

    # Load once at startup (expensive)
    ctx = load_models()

    # Call as many times as needed (fast)
    result = predict(perf_array, image_path, ctx)
    print(result["label_name"], result["confidence"])
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Add project root to sys.path so this module can be imported from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# CONFIGURATION  (mirrors fusion_model.py constants)
# ---------------------------------------------------------
LSTM_FEAT_PATH    = os.path.join(PROJECT_ROOT, "models", "saved_models", "lstm_features.h5")
CNN_FEAT_PATH     = os.path.join(PROJECT_ROOT, "models", "saved_models", "cnn_features.h5")
FUSION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "saved_models", "fusion_model.h5")

SEQUENCE_LENGTH  = 10
FEATURE_COLUMNS  = [
    "cpu_percent", "memory_percent", "process_count",
    "disk_read_bytes", "disk_write_bytes",
]
IMG_HEIGHT = 256
IMG_WIDTH  = 256

LABEL_NAMES = {0: "normal", 1: "suspicious"}

# Cached model context (populated by load_models())
_MODEL_CACHE: dict | None = None


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
def load_models(force_reload: bool = False) -> dict:
    """
    Load and cache all three models (lstm_features, cnn_features,
    fusion_model).  Subsequent calls return the cached version unless
    force_reload=True.

    Returns
    -------
    A context dict with keys:
        "lstm_extractor", "cnn_extractor", "fusion_model"
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    missing = [p for p in [LSTM_FEAT_PATH, CNN_FEAT_PATH, FUSION_MODEL_PATH]
               if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "The following model files are missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun fusion_model.py to generate them."
        )

    print("  Loading LSTM feature extractor...")
    lstm_ext = tf.keras.models.load_model(LSTM_FEAT_PATH, compile=False)

    print("  Loading CNN feature extractor...")
    cnn_ext  = tf.keras.models.load_model(CNN_FEAT_PATH,  compile=False)

    print("  Loading Fusion model...")
    fusion   = tf.keras.models.load_model(FUSION_MODEL_PATH, compile=False)

    # Warm up each model with a dummy forward pass so the first real
    # prediction is not artificially slow.
    _warmup_lstm(lstm_ext)
    _warmup_cnn(cnn_ext)
    _warmup_fusion(fusion, lstm_ext, cnn_ext)

    _MODEL_CACHE = {
        "lstm_extractor": lstm_ext,
        "cnn_extractor":  cnn_ext,
        "fusion_model":   fusion,
    }
    print("  All models loaded and warmed up.")
    return _MODEL_CACHE


def _warmup_lstm(model: tf.keras.Model) -> None:
    dummy = np.zeros((1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), dtype=np.float32)
    model(dummy, training=False)


def _warmup_cnn(model: tf.keras.Model) -> None:
    dummy = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    model(dummy, training=False)


def _warmup_fusion(fusion, lstm_ext, cnn_ext) -> None:
    lstm_feat = lstm_ext(
        np.zeros((1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), dtype=np.float32),
        training=False,
    ).numpy().flatten()
    cnn_feat = cnn_ext(
        np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32),
        training=False,
    ).numpy().flatten()
    fused = np.concatenate([lstm_feat, cnn_feat])[np.newaxis, :]
    fusion(fused.astype(np.float32), training=False)


# ---------------------------------------------------------
# INPUT PREPROCESSING
# ---------------------------------------------------------
def preprocess_perf_sequence(
    perf_data: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Normalise a performance sequence to a (1, SEQUENCE_LENGTH, 5) float32
    array ready for the LSTM feature extractor.

    Parameters
    ----------
    perf_data : array-like of shape (T, 5) where T >= SEQUENCE_LENGTH.
                Columns must be in FEATURE_COLUMNS order.
                If a DataFrame is passed, only FEATURE_COLUMNS are used.
    """
    if isinstance(perf_data, pd.DataFrame):
        avail = [c for c in FEATURE_COLUMNS if c in perf_data.columns]
        arr = perf_data[avail].values.astype(np.float64)
    else:
        arr = np.asarray(perf_data, dtype=np.float64)

    if arr.ndim == 1:
        raise ValueError(
            f"perf_data must be 2-D (timesteps x features), got shape {arr.shape}"
        )

    # Pad missing feature columns with 0 if fewer than 5 columns supplied
    if arr.shape[1] < len(FEATURE_COLUMNS):
        pad = np.zeros((arr.shape[0], len(FEATURE_COLUMNS) - arr.shape[1]),
                       dtype=np.float64)
        arr = np.concatenate([arr, pad], axis=1)

    if len(arr) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} timesteps, got {len(arr)}."
        )

    # Use the last SEQUENCE_LENGTH rows
    window = arr[-SEQUENCE_LENGTH:, :len(FEATURE_COLUMNS)].astype(np.float64)

    # Per-column min-max normalise
    for col_idx in range(window.shape[1]):
        col = window[:, col_idx]
        col_min, col_max = col.min(), col.max()
        if col_max - col_min > 0:
            window[:, col_idx] = (col - col_min) / (col_max - col_min)
        else:
            window[:, col_idx] = 0.0

    return window[np.newaxis, :, :].astype(np.float32)   # (1, 10, 5)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load a grayscale memory image and normalise to (1, 256, 256, 1) float32.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = load_img(image_path, color_mode="grayscale",
                   target_size=(IMG_HEIGHT, IMG_WIDTH))
    arr = img_to_array(img) / 255.0            # (256, 256, 1)
    return arr[np.newaxis, :, :, :].astype(np.float32)  # (1, 256, 256, 1)


# ---------------------------------------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------------------------------------
def predict(
    perf_data: np.ndarray | pd.DataFrame,
    image_path: str,
    ctx: dict | None = None,
) -> dict:
    """
    Run the full fusion inference pipeline.

    Parameters
    ----------
    perf_data  : array of shape (T, 5) or a DataFrame with FEATURE_COLUMNS.
                 T must be >= SEQUENCE_LENGTH (10).
    image_path : path to a 256x256 grayscale PNG.
    ctx        : model context from load_models().  If None, load_models()
                 is called automatically (adds ~2s on first call).

    Returns
    -------
    dict with keys:
        label         (int)   : 0 = normal, 1 = suspicious
        label_name    (str)   : "normal" or "suspicious"
        confidence    (float) : probability that the sample is suspicious
        lstm_features (ndarray shape 16) : LSTM branch feature vector
        cnn_features  (ndarray shape 128): CNN branch feature vector
        fused_vector  (ndarray shape 144): concatenated feature vector
    """
    if ctx is None:
        ctx = load_models()

    lstm_ext = ctx["lstm_extractor"]
    cnn_ext  = ctx["cnn_extractor"]
    fusion   = ctx["fusion_model"]

    # Preprocess inputs
    perf_seq = preprocess_perf_sequence(perf_data)   # (1, 10, 5)
    img_arr  = preprocess_image(image_path)           # (1, 256, 256, 1)

    # Extract features
    lstm_feat = lstm_ext(perf_seq, training=False).numpy().flatten()  # (16,)
    cnn_feat  = cnn_ext(img_arr,  training=False).numpy().flatten()   # (128,)
    fused     = np.concatenate([lstm_feat, cnn_feat])[np.newaxis, :]  # (1, 144)

    # Fuse and classify
    confidence = float(fusion(fused.astype(np.float32), training=False).numpy().flatten()[0])
    label      = int(confidence >= 0.5)

    return {
        "label":         label,
        "label_name":    LABEL_NAMES[label],
        "confidence":    round(confidence, 4),
        "lstm_features": lstm_feat,
        "cnn_features":  cnn_feat,
        "fused_vector":  fused.flatten(),
    }


# ---------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------
def _selftest() -> None:
    """
    Quick sanity-check using the existing session data.
    Loads the first normal image + recent performance rows and runs
    a full prediction.
    """
    import csv

    print("=" * 60)
    print("  HyperMD Inference -- Self-Test")
    print("=" * 60)

    # Load image_labels.csv to find a real image
    label_csv = os.path.join(PROJECT_ROOT, "memory_images", "image_labels.csv")
    if not os.path.isfile(label_csv):
        print("  ERROR: memory_images/image_labels.csv not found.")
        return

    rows = list(csv.DictReader(open(label_csv, encoding="utf-8")))
    if not rows:
        print("  ERROR: image_labels.csv is empty.")
        return

    # Use the first available image
    test_row = rows[0]
    image_path = os.path.join(PROJECT_ROOT, "memory_images", test_row["filename"])
    true_label = int(test_row["label"])

    print(f"\n  Test image : {test_row['filename']}")
    print(f"  True label : {true_label} ({LABEL_NAMES[true_label]})")

    # Load performance data
    perf_file = os.path.join(PROJECT_ROOT, "performance_data.csv")
    if not os.path.isfile(perf_file):
        print("  ERROR: performance_data.csv not found.")
        return

    perf_df = pd.read_csv(perf_file)
    avail = [c for c in FEATURE_COLUMNS if c in perf_df.columns]
    if len(avail) < 3 or len(perf_df) < SEQUENCE_LENGTH:
        print(f"  ERROR: Not enough performance data ({len(perf_df)} rows, "
              f"need {SEQUENCE_LENGTH}).")
        return

    print(f"  Using last {SEQUENCE_LENGTH} rows from performance_data.csv")

    print("\n  Loading models...")
    ctx = load_models()

    print("\n  Running prediction...")
    result = predict(perf_df, image_path, ctx)

    print()
    print(f"  Predicted label  : {result['label']} ({result['label_name']})")
    print(f"  True label       : {true_label} ({LABEL_NAMES[true_label]})")
    print(f"  Confidence       : {result['confidence']:.4f}  "
          f"({'correct' if result['label'] == true_label else 'INCORRECT'})")
    print(f"  LSTM feat shape  : {result['lstm_features'].shape}")
    print(f"  CNN feat shape   : {result['cnn_features'].shape}")
    print(f"  Fused vec shape  : {result['fused_vector'].shape}")
    print()
    print("  Self-test complete. Inference module is working correctly.")
    print("=" * 60)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HyperMD Inference Engine -- Module 4D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run a quick self-test using the first session's data.",
    )
    parser.add_argument(
        "--image", type=str, default="",
        help="Path to a PNG memory image to classify.",
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.image:
        perf_file = os.path.join(PROJECT_ROOT, "performance_data.csv")
        perf_df   = pd.read_csv(perf_file)
        ctx       = load_models()
        result    = predict(perf_df, args.image, ctx)
        print(f"Result: {result['label_name']}  "
              f"(confidence={result['confidence']:.4f})")
    else:
        parser.print_help()

