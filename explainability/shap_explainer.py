"""
HyperMD-Enhanced - Module 5A: SHAP Explainer (LSTM / Tabular Branch)
=====================================================================
Uses SHAP KernelExplainer to compute feature importance scores for the
LSTM branch of the fusion model.

KernelExplainer is model-agnostic and works on any Python function,
so it handles TensorFlow / Keras 3 on Windows without requiring GPU
or Linux-specific SHAP backends.

What it tells you
-----------------
For each prediction it answers: "Which performance feature (CPU, RAM,
process count, disk reads, disk writes) contributed most to classifying
this sequence as normal or suspicious?"

Output
------
A dict with:
  "shap_values"        : np.ndarray shape (5,)  -- mean |SHAP| per feature
  "feature_names"      : list of 5 feature names
  "top_features"       : list of (feature_name, shap_value) sorted by |value|
  "text_summary"       : human-readable string, e.g.
                         "Top drivers: cpu_percent (+0.42), memory_percent (+0.18)"

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\explainability\\shap_explainer.py

Usage
-----
    from explainability.shap_explainer import explain_lstm
    from models.inference import load_models
    import numpy as np

    ctx = load_models()
    result = explain_lstm(perf_sequence_array, ctx)
    print(result["text_summary"])
"""

import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FEATURE_COLUMNS = [
    "cpu_percent", "memory_percent", "process_count",
    "disk_read_bytes", "disk_write_bytes",
]
SEQUENCE_LENGTH = 10

# Number of background samples for KernelExplainer.
# More = better estimates, but slower.  50 is a good balance for 5 features.
N_BACKGROUND = 50
N_EXPLAIN    = 1   # explain one sequence at a time


def _lstm_predict_fn(ctx: dict):
    """
    Return a function that:
      - Takes a 2-D array of shape (N, SEQUENCE_LENGTH * num_features)
        (flat representation needed by SHAP)
      - Returns a 1-D array of shape (N,) with suspicious probabilities.

    SHAP requires the prediction function to accept a plain 2-D numpy array,
    so we reshape internally from flat -> (N, seq_len, features).
    """
    import tensorflow as tf
    lstm_ext  = ctx["lstm_extractor"]
    cnn_ext   = ctx["cnn_extractor"]
    fusion    = ctx["fusion_model"]

    # Dummy CNN feature vector (zeros) — we explain the LSTM branch only
    dummy_cnn = np.zeros((1, 128), dtype=np.float32)

    num_features = len(FEATURE_COLUMNS)
    flat_size    = SEQUENCE_LENGTH * num_features

    def predict(X_flat: np.ndarray) -> np.ndarray:
        """X_flat shape: (N, SEQUENCE_LENGTH * num_features)"""
        N = X_flat.shape[0]
        X_seq = X_flat.reshape(N, SEQUENCE_LENGTH, num_features).astype(np.float32)

        results = np.zeros(N, dtype=np.float32)
        for i in range(N):
            lstm_feat = lstm_ext(X_seq[i:i+1], training=False).numpy()  # (1, 16)
            cnn_feat  = dummy_cnn                                         # (1, 128)
            fused     = np.concatenate([lstm_feat, cnn_feat], axis=1)    # (1, 144)
            prob      = float(fusion(fused, training=False).numpy()[0, 0])
            results[i] = prob
        return results

    return predict, flat_size


def explain_lstm(
    perf_sequence: np.ndarray,
    ctx: dict,
    background_data: np.ndarray | None = None,
    use_shap: bool = False,
) -> dict:
    """
    Compute feature importance for one performance sequence.

    Parameters
    ----------
    perf_sequence  : np.ndarray shape (SEQUENCE_LENGTH, num_features)
                     Already preprocessed (scaled 0-1).
    ctx            : model context from models.inference.load_models()
    background_data: optional array for SHAP background (ignored if use_shap=False)
    use_shap       : if True, use SHAP KernelExplainer (accurate but slow ~3-5 min).
                     if False (default), use gradient-based importance (fast, <1s).

    Returns
    -------
    dict with keys:
        shap_values   (ndarray shape 5) : mean absolute importance per feature
        feature_names (list of 5 str)
        top_features  (list of tuples (name, value) sorted by |value| desc)
        text_summary  (str)
    """
    if not use_shap:
        return _shap_unavailable_fallback(perf_sequence, ctx)

    try:
        import shap
    except ImportError:
        return _shap_unavailable_fallback(perf_sequence, ctx)

    num_features = len(FEATURE_COLUMNS)
    seq_len      = perf_sequence.shape[0]
    flat_size    = seq_len * num_features

    predict_fn, _ = _lstm_predict_fn(ctx)

    # Background data: SHAP measures "change from background"
    if background_data is not None:
        bg = background_data.reshape(len(background_data), flat_size).astype(np.float32)
    else:
        # Neutral background: all zeros (represents "idle" state)
        bg = np.zeros((N_BACKGROUND, flat_size), dtype=np.float32)

    # Explainer and values
    explainer = shap.KernelExplainer(predict_fn, bg, silent=True)
    x_flat    = perf_sequence.flatten()[np.newaxis, :]  # (1, flat_size)

    # nsamples controls approximation quality; 100 is fast enough
    shap_vals = explainer.shap_values(x_flat, nsamples=50, silent=True)
    # shap_vals shape: (1, flat_size)

    # Aggregate over time steps: mean |SHAP| per feature column
    shap_matrix = shap_vals[0].reshape(seq_len, num_features)  # (10, 5)
    mean_shap   = np.abs(shap_matrix).mean(axis=0)             # (5,)

    return _build_result(mean_shap)


def _shap_unavailable_fallback(perf_sequence: np.ndarray, ctx: dict) -> dict:
    """
    If SHAP is not installed, use gradient-based feature importance
    as an approximate substitute (no extra dependencies).
    """
    import tensorflow as tf
    lstm_ext = ctx["lstm_extractor"]

    seq = tf.constant(perf_sequence[np.newaxis, :, :], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(seq)
        output = lstm_ext(seq, training=False)
        score  = tf.reduce_sum(output)

    grads = tape.gradient(score, seq).numpy()[0]   # (10, 5)
    mean_shap = np.abs(grads).mean(axis=0)         # (5,)
    return _build_result(mean_shap)


def _build_result(mean_importance: np.ndarray) -> dict:
    pairs     = list(zip(FEATURE_COLUMNS, mean_importance.tolist()))
    sorted_p  = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    top2      = sorted_p[:2]

    parts = []
    for name, val in top2:
        sign = "+" if val >= 0 else "-"
        parts.append(f"{name} ({sign}{abs(val):.3f})")
    text = "Top drivers: " + ", ".join(parts) if parts else "No significant drivers."

    return {
        "shap_values":   mean_importance,
        "feature_names": FEATURE_COLUMNS,
        "top_features":  sorted_p,
        "text_summary":  text,
    }

