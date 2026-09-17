"""
HyperMD-Enhanced - Module 5C: Unified Explanation Generator
============================================================
Combines SHAP (LSTM branch) and Grad-CAM (CNN branch) into one
human-readable explanation for every fusion model prediction.

This is the module called by the dashboard and response engine to
produce the "why" alongside every alert.

Output example
--------------
  "PREDICTION: suspicious (confidence 0.89)
   LSTM (time-series) drivers:
     #1 cpu_percent       +0.423  [HIGH IMPACT]
     #2 memory_percent    +0.187
     #3 disk_write_bytes  +0.042
   CNN (memory image) driver:
     Hottest region: upper-right quadrant
     (rows 0-64, cols 128-192) -- approx. offset 0x00002000
   SUMMARY: Flagged due to sustained high CPU (SHAP rank 1) and
   anomalous memory texture in the upper-right region (Grad-CAM hotspot)."

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\explainability\\explain.py

Usage
-----
    .venv\\Scripts\\python.exe explainability/explain.py --selftest

    from explainability.explain import explain_prediction
    from models.inference import load_models, predict, preprocess_perf_sequence

    ctx = load_models()
    result = predict(perf_df, image_path, ctx)
    explanation = explain_prediction(
        perf_sequence  = preprocess_perf_sequence(perf_df)[0],
        image_path     = image_path,
        predict_result = result,
        ctx            = ctx,
    )
    print(explanation["full_text"])
"""

import os
import sys
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from explainability.shap_explainer import explain_lstm
from explainability.gradcam         import explain_cnn

LABEL_NAMES    = {0: "normal", 1: "suspicious"}
FEATURE_LABELS = {
    "cpu_percent":       "CPU usage",
    "memory_percent":    "RAM usage",
    "process_count":     "Process count",
    "disk_read_bytes":   "Disk read activity",
    "disk_write_bytes":  "Disk write activity",
}
EXPLANATION_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs", "explanations")


def explain_prediction(
    perf_sequence: np.ndarray,
    image_path: str,
    predict_result: dict,
    ctx: dict,
    save_gradcam: bool = True,
    event_id: str = "",
) -> dict:
    """
    Generate a full multi-modal explanation for one prediction.

    Parameters
    ----------
    perf_sequence  : np.ndarray shape (SEQUENCE_LENGTH, 5)
                     Already preprocessed (min-max scaled 0-1).
    image_path     : path to the 256x256 grayscale PNG.
    predict_result : output dict from models.inference.predict()
    ctx            : model context from models.inference.load_models()
    save_gradcam   : if True, save the Grad-CAM overlay PNG to logs/explanations/
    event_id       : optional tag for the overlay filename

    Returns
    -------
    dict with keys:
        label          (int)
        label_name     (str)
        confidence     (float)
        shap_result    (dict from shap_explainer.explain_lstm)
        gradcam_result (dict from gradcam.explain_cnn)
        full_text      (str) -- complete human-readable explanation
        short_summary  (str) -- one-line summary for dashboard cards
        overlay_path   (str) -- path to saved Grad-CAM overlay (or "")
    """
    label       = predict_result["label"]
    confidence  = predict_result["confidence"]
    label_name  = LABEL_NAMES[label]

    # ── SHAP explanation (LSTM branch) ──────────────────────────────────────
    shap_result = explain_lstm(perf_sequence, ctx)

    # ── Grad-CAM explanation (CNN branch) ───────────────────────────────────
    overlay_path = ""
    if save_gradcam and event_id:
        os.makedirs(EXPLANATION_OUTPUT_DIR, exist_ok=True)
        overlay_path = os.path.join(
            EXPLANATION_OUTPUT_DIR,
            f"{event_id}_gradcam_overlay.png"
        )

    gradcam_result = explain_cnn(image_path, ctx, save_overlay_path=overlay_path)

    # ── Compose full text explanation ───────────────────────────────────────
    full_text  = _compose_full_text(
        label_name, confidence, shap_result, gradcam_result
    )
    short_summary = _compose_short_summary(
        label_name, confidence, shap_result, gradcam_result
    )

    return {
        "label":          label,
        "label_name":     label_name,
        "confidence":     confidence,
        "shap_result":    shap_result,
        "gradcam_result": gradcam_result,
        "full_text":      full_text,
        "short_summary":  short_summary,
        "overlay_path":   overlay_path,
    }


def _compose_full_text(
    label_name: str,
    confidence: float,
    shap_result: dict,
    gradcam_result: dict,
) -> str:
    lines = []
    lines.append(
        f"PREDICTION: {label_name.upper()} "
        f"(confidence {confidence:.2%})"
    )
    lines.append("-" * 50)

    # SHAP section
    lines.append("LSTM (time-series) feature importance:")
    top = shap_result["top_features"]
    for rank, (feat, val) in enumerate(top, start=1):
        human_name = FEATURE_LABELS.get(feat, feat)
        bar        = _bar(abs(val))
        impact_tag = " [HIGH IMPACT]" if rank == 1 else ""
        lines.append(
            f"  #{rank:1d}  {human_name:<22s}  {val:+.4f}  {bar}{impact_tag}"
        )

    lines.append("")
    # Grad-CAM section
    lines.append("CNN (memory image) spatial attention:")
    lines.append(f"  {gradcam_result['hotspot_desc']}")

    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"  {_compose_short_summary(label_name, confidence, shap_result, gradcam_result)}")

    return "\n".join(lines)


def _compose_short_summary(
    label_name: str,
    confidence: float,
    shap_result: dict,
    gradcam_result: dict,
) -> str:
    top    = shap_result["top_features"]
    top1_n = FEATURE_LABELS.get(top[0][0], top[0][0]) if top else "unknown"
    top2_n = FEATURE_LABELS.get(top[1][0], top[1][0]) if len(top) > 1 else ""

    heatmap     = gradcam_result.get("heatmap")
    max_act     = float(heatmap.max()) if heatmap is not None else 0.0
    hotspot_txt = gradcam_result.get("hotspot_desc", "").split(".")[0]

    if label_name == "suspicious":
        lstm_part   = f"high {top1_n} (SHAP rank 1)"
        if top2_n:
            lstm_part += f" and elevated {top2_n} (rank 2)"
        return (
            f"Flagged as suspicious ({confidence:.0%} confidence) due to "
            f"{lstm_part}; memory image shows anomalous texture "
            f"({hotspot_txt}, intensity {max_act:.2f})."
        )
    else:
        return (
            f"Classified as normal ({confidence:.0%} confidence). "
            f"No significant LSTM anomalies. "
            f"Memory image texture intensity: {max_act:.2f}."
        )


def _bar(value: float, width: int = 10) -> str:
    """ASCII progress bar for SHAP magnitude."""
    filled = int(round(value * width / max(value, 1e-6)))
    filled = min(filled, width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ---------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------
def _selftest() -> None:
    import csv
    import pandas as pd

    print("=" * 62)
    print("  HyperMD Explainability -- Self-Test")
    print("=" * 62)

    # Import inference utilities
    from models.inference import (
        load_models, predict, preprocess_perf_sequence
    )

    # Find a real image
    label_csv = os.path.join(PROJECT_ROOT, "memory_images", "image_labels.csv")
    if not os.path.isfile(label_csv):
        print("  ERROR: image_labels.csv not found.")
        return

    rows = list(csv.DictReader(open(label_csv, encoding="utf-8")))
    if not rows:
        print("  ERROR: image_labels.csv is empty.")
        return

    test_row   = rows[0]
    image_path = os.path.join(PROJECT_ROOT, "memory_images", test_row["filename"])
    if not os.path.isfile(image_path):
        print(f"  ERROR: Image not found: {image_path}")
        return

    # Load performance data
    perf_file = os.path.join(PROJECT_ROOT, "performance_data.csv")
    perf_df   = pd.read_csv(perf_file)

    print(f"\n  Test image  : {test_row['filename']}  (label={test_row['label']})")
    print(f"  Perf rows   : {len(perf_df)}")

    print("\n  Loading models...")
    ctx = load_models()

    print("  Running predict()...")
    result = predict(perf_df, image_path, ctx)

    print("  Computing explanation...")
    perf_seq = preprocess_perf_sequence(perf_df)[0]  # (10, 5)

    explanation = explain_prediction(
        perf_sequence  = perf_seq,
        image_path     = image_path,
        predict_result = result,
        ctx            = ctx,
        save_gradcam   = True,
        event_id       = test_row.get("event_id", "selftest"),
    )

    print()
    print(explanation["full_text"])
    print()
    if explanation["overlay_path"]:
        print(f"  Grad-CAM overlay saved -> {explanation['overlay_path']}")
    print()
    print("  Self-test complete.")
    print("=" * 62)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HyperMD Explainability -- Module 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run self-test using first session image + recent perf data.",
    )
    parser.add_argument(
        "--image", default="",
        help="Path to PNG memory image to explain.",
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
    elif args.image:
        import pandas as pd
        from models.inference import load_models, predict, preprocess_perf_sequence

        perf_df = pd.read_csv(os.path.join(PROJECT_ROOT, "performance_data.csv"))
        ctx     = load_models()
        result  = predict(perf_df, args.image, ctx)
        perf_seq = preprocess_perf_sequence(perf_df)[0]
        exp     = explain_prediction(
            perf_sequence  = perf_seq,
            image_path     = args.image,
            predict_result = result,
            ctx            = ctx,
            save_gradcam   = True,
        )
        print(exp["full_text"])
    else:
        parser.print_help()

