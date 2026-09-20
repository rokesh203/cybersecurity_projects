"""
HyperMD-Enhanced - Module 8: Model Evaluation
==============================================
Evaluates all three trained models (LSTM, CNN, Fusion) against the
collected session data, prints a comprehensive report, and saves plots.

Metrics computed
----------------
  - Accuracy, Precision, Recall, F1 (per class + macro/weighted)
  - Confusion matrix
  - ROC curve + AUC
  - PR (Precision-Recall) curve + AP

Reports saved
-------------
  logs/evaluation/
    evaluation_report.txt       -- full text report
    lstm_confusion.png          -- confusion matrix heatmap
    cnn_confusion.png
    fusion_confusion.png
    roc_curves.png              -- all three ROC curves on one plot
    pr_curves.png               -- all three PR curves on one plot

Run
---
    .venv\\Scripts\\python.exe evaluation\\evaluate.py

Requirements
------------
  - models/saved_models/lstm_model.h5      (trained)
  - models/saved_models/cnn_model.h5       (trained)
  - models/saved_models/fusion_model.h5    (trained)
  - models/saved_models/lstm_features.h5   (feature extractor)
  - models/saved_models/cnn_features.h5    (feature extractor)
  - performance_data.csv
  - memory_images/image_labels.csv
  - event_registry.csv
"""

import os
import sys
import csv
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from PIL import Image

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
LSTM_MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "saved_models", "lstm_model.h5")
CNN_MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "saved_models", "cnn_model.h5")
FUSION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "saved_models", "fusion_model.h5")
LSTM_FEAT_PATH    = os.path.join(PROJECT_ROOT, "models", "saved_models", "lstm_features.h5")
CNN_FEAT_PATH     = os.path.join(PROJECT_ROOT, "models", "saved_models", "cnn_features.h5")

PERFORMANCE_FILE  = os.path.join(PROJECT_ROOT, "performance_data.csv")
IMAGE_LABELS_CSV  = os.path.join(PROJECT_ROOT, "memory_images", "image_labels.csv")
EVENT_REGISTRY    = os.path.join(PROJECT_ROOT, "event_registry.csv")
MEMORY_IMAGES_DIR = os.path.join(PROJECT_ROOT, "memory_images")

EVAL_DIR          = os.path.join(PROJECT_ROOT, "logs", "evaluation")

SEQUENCE_LENGTH  = 10
FEATURE_COLUMNS  = [
    "cpu_percent", "memory_percent", "process_count",
    "disk_read_bytes", "disk_write_bytes",
]
IMG_SIZE = 256


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------
def _scale_perf(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0
    return df


def load_lstm_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Build sequences from performance_data.csv.
    Returns (X, y) where X.shape=(N, 10, 5), y.shape=(N,).
    Rows without a label (-1) are excluded.
    """
    if not os.path.isfile(PERFORMANCE_FILE):
        return np.empty((0,)), np.empty((0,))

    df = pd.read_csv(PERFORMANCE_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Only use tagged rows (event_id not empty, label != -1)
    if "label" in df.columns:
        df = df[df["label"].isin([0, 1])]
    else:
        print("  WARNING: performance_data.csv has no 'label' column. "
              "LSTM evaluation will use rule-based labels.")
        df["label"] = (df["cpu_percent"] > 80).astype(int)

    if len(df) < SEQUENCE_LENGTH + 1:
        return np.empty((0,)), np.empty((0,))

    df = _scale_perf(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    data   = df[FEATURE_COLUMNS].values
    labels = df["label"].values

    X, y = [], []
    for i in range(len(data) - SEQUENCE_LENGTH):
        X.append(data[i: i + SEQUENCE_LENGTH])
        y.append(int(labels[i + SEQUENCE_LENGTH - 1]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_cnn_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load all images from image_labels.csv.
    Returns (X, y) where X.shape=(N, 256, 256, 1), y.shape=(N,).
    """
    if not os.path.isfile(IMAGE_LABELS_CSV):
        return np.empty((0,)), np.empty((0,))

    rows = list(csv.DictReader(open(IMAGE_LABELS_CSV, encoding="utf-8")))
    X, y = [], []
    for row in rows:
        img_path = os.path.join(MEMORY_IMAGES_DIR, row["filename"])
        if not os.path.isfile(img_path):
            continue
        img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        X.append(arr[:, :, np.newaxis])
        y.append(int(row["label"]))

    if not X:
        return np.empty((0,)), np.empty((0,))
    return np.array(X), np.array(y, dtype=np.int32)


def load_fusion_dataset(
    lstm_ext: tf.keras.Model,
    cnn_ext:  tf.keras.Model,
    events:   list[dict],
    perf_df:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (fused_vector, label) pairs from event_registry.csv.
    """
    X, y = [], []
    BUFFER = pd.Timedelta(seconds=60)

    for event in events:
        t_start = pd.to_datetime(event["perf_start"]) - BUFFER
        t_end   = pd.to_datetime(event["perf_end"])   + BUFFER
        mask    = (perf_df["timestamp"] >= t_start) & (perf_df["timestamp"] <= t_end)
        window  = perf_df.loc[mask].copy()

        if len(window) < SEQUENCE_LENGTH:
            window = perf_df.copy()
        if len(window) < SEQUENCE_LENGTH:
            continue

        window = _scale_perf(window)
        for col in FEATURE_COLUMNS:
            if col not in window.columns:
                window[col] = 0.0

        data      = window[FEATURE_COLUMNS].values
        sequences = [data[i:i + SEQUENCE_LENGTH]
                     for i in range(len(data) - SEQUENCE_LENGTH)]
        if not sequences:
            continue

        X_seq     = np.array(sequences, dtype=np.float32)
        lstm_feat = lstm_ext.predict(X_seq, verbose=0).mean(axis=0)

        img_path = os.path.join(MEMORY_IMAGES_DIR, event["image_file"])
        if not os.path.isfile(img_path):
            continue
        img      = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img_arr  = np.array(img, dtype=np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
        cnn_feat = cnn_ext.predict(img_arr, verbose=0).flatten()

        fused = np.concatenate([lstm_feat, cnn_feat])
        X.append(fused)
        y.append(int(event["label"]))

    if not X:
        return np.empty((0,)), np.empty((0,))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------
# PLOTTING (pure PIL/NumPy — no matplotlib needed)
# ---------------------------------------------------------
def _save_confusion_png(cm: np.ndarray, title: str, path: str) -> None:
    """Save a simple confusion matrix as a PNG using PIL."""
    cell = 80
    pad  = 40
    n    = cm.shape[0]
    W    = cell * n + pad * 2
    H    = cell * n + pad * 2 + 30

    img = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    max_val = cm.max() if cm.max() > 0 else 1

    for r in range(n):
        for c in range(n):
            val   = cm[r, c]
            intensity = int(255 * (1 - val / max_val))
            color = (intensity, intensity, 255) if r == c else (255, intensity, intensity)
            x0 = pad + c * cell
            y0 = pad + 30 + r * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=color, outline="gray")
            draw.text((x0 + cell // 2 - 10, y0 + cell // 2 - 8),
                      str(val), fill="black")

    draw.text((pad, 5), title, fill="black")
    labels = ["Normal", "Suspicious"]
    for i, lbl in enumerate(labels[:n]):
        draw.text((pad + i * cell + 10, pad + 8), lbl[:6], fill="navy")
        draw.text((2, pad + 30 + i * cell + 30), lbl[:6], fill="navy")

    img.save(path)


def _save_roc_png(roc_data: list[dict], path: str) -> None:
    """Save ROC curves as a PNG using PIL drawing."""
    W, H   = 400, 400
    margin = 60
    img    = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw
    draw   = ImageDraw.Draw(img)

    # Axes
    draw.line([(margin, margin), (margin, H - margin)], fill="black", width=2)
    draw.line([(margin, H - margin), (W - margin, H - margin)], fill="black", width=2)
    draw.text((2, H // 2), "TPR", fill="black")
    draw.text((W // 2, H - 20), "FPR", fill="black")
    draw.text((margin + 5, 5), "ROC Curves", fill="black")

    # Diagonal
    pw = W - 2 * margin
    ph = H - 2 * margin
    draw.line([(margin, H - margin), (W - margin, margin)],
              fill="lightgray", width=1)

    colors = ["blue", "red", "green"]
    for idx, entry in enumerate(roc_data):
        fpr, tpr, roc_auc = entry["fpr"], entry["tpr"], entry["auc"]
        name  = entry["name"]
        color = colors[idx % len(colors)]
        pts   = []
        for f, t in zip(fpr, tpr):
            x = int(margin + f * pw)
            y = int(H - margin - t * ph)
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
        draw.text((margin + 5, margin + 15 * idx + 5),
                  f"{name}: AUC={roc_auc:.3f}", fill=color)

    img.save(path)


# ---------------------------------------------------------
# EVALUATION CORE
# ---------------------------------------------------------
def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all metrics for one model."""
    if len(y_true) == 0:
        return {"name": name, "error": "no data"}

    y_pred = (y_pred_prob >= threshold).astype(int)

    report = classification_report(
        y_true, y_pred,
        target_names=["normal", "suspicious"],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    roc_data = {}
    pr_data  = {}
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc     = auc(fpr, tpr)
        prec, rec, _= precision_recall_curve(y_true, y_pred_prob)
        ap          = average_precision_score(y_true, y_pred_prob)
        roc_data    = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        pr_data     = {"precision": prec, "recall": rec, "ap": ap}
    else:
        roc_data = {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": 0.5}
        pr_data  = {"ap": 0.0}

    return {
        "name":       name,
        "n_samples":  len(y_true),
        "report":     report,
        "cm":         cm,
        "roc":        roc_data,
        "pr":         pr_data,
        "accuracy":   report["accuracy"],
    }


# ---------------------------------------------------------
# REPORT WRITING
# ---------------------------------------------------------
def _write_report(results: list[dict], report_path: str) -> None:
    lines = []
    lines.append("=" * 64)
    lines.append("  HyperMD-Enhanced -- Model Evaluation Report")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 64)

    for res in results:
        lines.append(f"\n{'='*64}")
        lines.append(f"  Model: {res['name']}")
        lines.append(f"  Samples: {res.get('n_samples', 0)}")
        lines.append(f"{'='*64}")

        if "error" in res:
            lines.append(f"  ERROR: {res['error']}")
            continue

        lines.append(f"  Accuracy : {res['accuracy']:.4f}")

        roc = res.get("roc", {})
        if "auc" in roc:
            lines.append(f"  ROC AUC  : {roc['auc']:.4f}")

        pr = res.get("pr", {})
        if "ap" in pr:
            lines.append(f"  Avg Prec : {pr['ap']:.4f}")

        lines.append("")
        lines.append("  Classification Report:")
        report = res.get("report", {})
        for cls in ["normal", "suspicious", "macro avg", "weighted avg"]:
            if cls in report:
                r = report[cls]
                if isinstance(r, dict):
                    lines.append(
                        f"    {cls:<16s}  "
                        f"P={r.get('precision', 0):.3f}  "
                        f"R={r.get('recall', 0):.3f}  "
                        f"F1={r.get('f1-score', 0):.3f}  "
                        f"N={int(r.get('support', 0))}"
                    )

        cm = res.get("cm")
        if cm is not None:
            lines.append("")
            lines.append("  Confusion Matrix:")
            lines.append("              Pred:Normal  Pred:Susp")
            for i, row_label in enumerate(["True:Normal", "True:Susp "]):
                lines.append(f"    {row_label}   {cm[i, 0]:>10d}  {cm[i, 1]:>9d}")

    lines.append("\n" + "=" * 64)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  Report saved -> {report_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("  HyperMD-Enhanced -- Evaluation (Module 8)")
    print("=" * 64)

    os.makedirs(EVAL_DIR, exist_ok=True)
    all_results = []
    roc_list    = []

    # ── Load base models ──────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    missing = [p for p in [LSTM_MODEL_PATH, CNN_MODEL_PATH, FUSION_MODEL_PATH,
                            LSTM_FEAT_PATH, CNN_FEAT_PATH]
               if not os.path.isfile(p)]
    if missing:
        print("  ERROR: Missing model files:")
        for m in missing:
            print(f"    {m}")
        return

    lstm_model  = tf.keras.models.load_model(LSTM_MODEL_PATH,  compile=False)
    cnn_model   = tf.keras.models.load_model(CNN_MODEL_PATH,   compile=False)
    fusion_model= tf.keras.models.load_model(FUSION_MODEL_PATH,compile=False)
    lstm_ext    = tf.keras.models.load_model(LSTM_FEAT_PATH,   compile=False)
    cnn_ext     = tf.keras.models.load_model(CNN_FEAT_PATH,    compile=False)
    print("  All models loaded.")

    # Warm up
    dummy_seq = np.zeros((1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), dtype=np.float32)
    dummy_img = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    lstm_model(dummy_seq); cnn_model(dummy_img)
    lstm_ext(dummy_seq);   cnn_ext(dummy_img)

    # ── LSTM evaluation ───────────────────────────────────────────────
    print("\n[2/4] Evaluating LSTM model...")
    X_lstm, y_lstm = load_lstm_dataset()
    if len(X_lstm) > 0:
        probs_lstm = lstm_model.predict(X_lstm, verbose=0).flatten()
        res_lstm   = evaluate_model("LSTM (Time-Series)", y_lstm, probs_lstm)
        all_results.append(res_lstm)
        if "roc" in res_lstm:
            roc_list.append({"name": "LSTM", **res_lstm["roc"]})
        _save_confusion_png(
            res_lstm["cm"],
            "LSTM Confusion Matrix",
            os.path.join(EVAL_DIR, "lstm_confusion.png")
        )
        print(f"  LSTM: {len(X_lstm)} sequences | "
              f"accuracy={res_lstm['accuracy']:.3f} | "
              f"AUC={res_lstm['roc']['auc']:.3f}")
    else:
        print("  WARNING: No LSTM data available.")
        all_results.append({"name": "LSTM (Time-Series)", "error": "insufficient data"})

    # ── CNN evaluation ────────────────────────────────────────────────
    print("\n[3/4] Evaluating CNN model...")
    X_cnn, y_cnn = load_cnn_dataset()
    if len(X_cnn) > 0:
        probs_cnn = cnn_model.predict(X_cnn, verbose=0).flatten()
        res_cnn   = evaluate_model("CNN (Memory Image)", y_cnn, probs_cnn)
        all_results.append(res_cnn)
        if "roc" in res_cnn:
            roc_list.append({"name": "CNN", **res_cnn["roc"]})
        _save_confusion_png(
            res_cnn["cm"],
            "CNN Confusion Matrix",
            os.path.join(EVAL_DIR, "cnn_confusion.png")
        )
        print(f"  CNN: {len(X_cnn)} images | "
              f"accuracy={res_cnn['accuracy']:.3f} | "
              f"AUC={res_cnn['roc']['auc']:.3f}")
    else:
        print("  WARNING: No CNN image data available.")
        all_results.append({"name": "CNN (Memory Image)", "error": "no images found"})

    # ── Fusion evaluation ─────────────────────────────────────────────
    print("\n[4/4] Evaluating Fusion model...")
    if not os.path.isfile(EVENT_REGISTRY):
        print("  WARNING: event_registry.csv not found.")
        all_results.append({"name": "Fusion Model", "error": "no event registry"})
    else:
        events = [r for r in csv.DictReader(open(EVENT_REGISTRY, encoding="utf-8"))
                  if r.get("image_file", "").strip()]

        perf_df = pd.read_csv(PERFORMANCE_FILE) if os.path.isfile(PERFORMANCE_FILE) else pd.DataFrame()
        if not perf_df.empty:
            perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"], errors="coerce")
            perf_df = perf_df.dropna(subset=["timestamp"])

        X_fus, y_fus = load_fusion_dataset(lstm_ext, cnn_ext, events, perf_df)
        if len(X_fus) > 0:
            probs_fus = fusion_model.predict(X_fus, verbose=0).flatten()
            res_fus   = evaluate_model("Fusion Model", y_fus, probs_fus)
            all_results.append(res_fus)
            if "roc" in res_fus:
                roc_list.append({"name": "Fusion", **res_fus["roc"]})
            _save_confusion_png(
                res_fus["cm"],
                "Fusion Confusion Matrix",
                os.path.join(EVAL_DIR, "fusion_confusion.png")
            )
            print(f"  Fusion: {len(X_fus)} matched pairs | "
                  f"accuracy={res_fus['accuracy']:.3f} | "
                  f"AUC={res_fus['roc']['auc']:.3f}")
        else:
            print("  WARNING: Not enough matched pairs for fusion evaluation.")
            all_results.append({"name": "Fusion Model", "error": "insufficient matched pairs"})

    # ── Save ROC plot ─────────────────────────────────────────────────
    if roc_list:
        _save_roc_png(roc_list, os.path.join(EVAL_DIR, "roc_curves.png"))
        print(f"\n  ROC plot saved -> {os.path.join(EVAL_DIR, 'roc_curves.png')}")

    # ── Write text report ─────────────────────────────────────────────
    _write_report(all_results, os.path.join(EVAL_DIR, "evaluation_report.txt"))

    print("\n" + "=" * 64)
    print(f"  All evaluation outputs -> {EVAL_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()

