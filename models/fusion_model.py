"""
HyperMD-Enhanced - Module 4C: Multi-Modal Late-Fusion Model
============================================================
Combines the LSTM (time-series performance) branch and the CNN
(memory-image) branch into a single fused classifier.

Architecture (Late Fusion)
--------------------------

  LSTM model (frozen)
    Input : sequence (10 timesteps x 5 features)
    Output: Dense(16, relu) penultimate layer  --> 16-dim feature vector
                                                        |
                                                   Concatenate  --> 144-dim
                                                        |
  CNN model (frozen)                             Dense(64, relu)
    Input : 256x256 grayscale image                    |
    Output: Dense(128, relu) "dense_features"    Dropout(0.3)
            --> 128-dim feature vector                  |
                                                   Dense(1, sigmoid)

Training data
-------------
Matched pairs from event_registry.csv:
  Each row pairs one performance window with one memory image,
  both tagged with the same event_id and label.

For each matched event:
  - LSTM branch: extract features from every valid 10-step sequence
    in that event's performance window, then AVERAGE them.
  - CNN branch: extract the 128-dim feature vector from the event's image.
  - Result: one (144,) feature vector per event.

Because we start with very few events, the training data is augmented
with Gaussian noise (sigma=0.01) to improve generalisation.

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\models\\fusion_model.py

Run
---
    .venv\\Scripts\\python.exe models/fusion_model.py

Expected output
---------------
  - Feature extraction log for each matched event
  - Training accuracy for fusion head (20 epochs)
  - Saved: models/saved_models/fusion_model.h5
  - Saved: models/saved_models/lstm_features.h5
  - Saved: models/saved_models/cnn_features.h5
"""

import os
import csv
import sys
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path for sibling imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
LSTM_MODEL_PATH    = "models/saved_models/lstm_model.h5"
CNN_MODEL_PATH     = "models/saved_models/cnn_model.h5"
FUSION_MODEL_PATH  = "models/saved_models/fusion_model.h5"
LSTM_FEAT_PATH     = "models/saved_models/lstm_features.h5"
CNN_FEAT_PATH      = "models/saved_models/cnn_features.h5"

EVENT_REGISTRY     = "event_registry.csv"
PERFORMANCE_FILE   = "performance_data.csv"
IMAGE_FOLDER       = "memory_images"

SEQUENCE_LENGTH    = 10
FEATURE_COLUMNS    = [
    "cpu_percent", "memory_percent", "process_count",
    "disk_read_bytes", "disk_write_bytes",
]

IMG_HEIGHT  = 256
IMG_WIDTH   = 256
EPOCHS      = 30
BATCH_SIZE  = 4
AUGMENT_N   = 20    # noise-augmented copies per real sample
NOISE_SIGMA = 0.02  # std-dev of Gaussian noise for augmentation


# ---------------------------------------------------------
# STEP 1 — BUILD FEATURE EXTRACTORS
# ---------------------------------------------------------
def build_lstm_extractor(lstm_model: tf.keras.Model) -> tf.keras.Model:
    """
    Create a sub-model that outputs the penultimate Dense layer
    of the trained LSTM model.  We freeze all weights.

    Works with TensorFlow 2.x including >= 2.16 where
    layer.output_shape was removed.
    """
    # Walk layers in reverse; first Dense whose output units != 1 is the
    # penultimate feature layer.
    target_layer = None
    for layer in reversed(lstm_model.layers):
        if not isinstance(layer, layers.Dense):
            continue
        # Get output units from the layer config (always available)
        units = layer.get_config().get("units", None)
        if units is not None and units != 1:
            target_layer = layer
            break

    if target_layer is None:
        raise ValueError(
            "Could not find a non-output Dense layer in the LSTM model. "
            "Check lstm_model.py architecture."
        )

    units = target_layer.get_config()["units"]
    print(f"  LSTM feature layer: '{target_layer.name}'  units={units}")

    # Call the model on a dummy input to force initialisation
    # (needed when loading Sequential models from .h5 in Keras 3)
    dummy = np.zeros((1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), dtype=np.float32)
    lstm_model(dummy)

    extractor = tf.keras.Model(
        inputs=lstm_model.inputs,
        outputs=target_layer.output,
        name="lstm_feature_extractor",
    )
    extractor.trainable = False
    return extractor


def build_cnn_extractor(cnn_model: tf.keras.Model) -> tf.keras.Model:
    """
    Create a sub-model that outputs the 'dense_features' layer
    of the trained CNN model.  We freeze all weights.
    """
    target_layer = cnn_model.get_layer("dense_features")
    print(f"  CNN feature layer : 'dense_features'  units=128")

    # Call the model on a dummy input to force initialisation
    dummy = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    cnn_model(dummy)

    extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=target_layer.output,
        name="cnn_feature_extractor",
    )
    extractor.trainable = False
    return extractor


# ---------------------------------------------------------
# STEP 2 — LOAD MATCHED PAIRS FROM EVENT REGISTRY
# ---------------------------------------------------------
def load_event_registry() -> list[dict]:
    """
    Read event_registry.csv and return a list of dicts, one per event.
    Filters out events with no image file.
    """
    if not os.path.isfile(EVENT_REGISTRY):
        raise FileNotFoundError(
            f"{EVENT_REGISTRY} not found. Run event_session.py first "
            "to create labeled sessions."
        )

    events = []
    with open(EVENT_REGISTRY, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("image_file", "").strip():
                print(f"  Skipping event {row['event_id']} — no image file recorded.")
                continue
            img_path = os.path.join(IMAGE_FOLDER, row["image_file"])
            if not os.path.isfile(img_path):
                print(f"  Skipping event {row['event_id']} — image not found: {img_path}")
                continue
            events.append(row)

    print(f"  Found {len(events)} matched events in {EVENT_REGISTRY}")
    return events


# ---------------------------------------------------------
# STEP 3 — EXTRACT FEATURES PER EVENT
# ---------------------------------------------------------
def _scale_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Min-max scale the given columns in-place (0–1 range per column)."""
    for col in cols:
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0
    return df


def extract_lstm_features(
    event: dict,
    perf_df: pd.DataFrame,
    lstm_extractor: tf.keras.Model,
) -> np.ndarray | None:
    """
    For a given event, select the performance rows in its time window,
    build overlapping sequences of length SEQUENCE_LENGTH, run them
    through the LSTM feature extractor, and return the MEAN feature vector.
    Returns None if there are insufficient rows.
    """
    BUFFER = pd.Timedelta(seconds=60)
    t_start = pd.to_datetime(event["perf_start"]) - BUFFER
    t_end   = pd.to_datetime(event["perf_end"])   + BUFFER

    mask = (perf_df["timestamp"] >= t_start) & (perf_df["timestamp"] <= t_end)
    window = perf_df.loc[mask].copy()

    if len(window) < SEQUENCE_LENGTH:
        print(f"  [LSTM] event {event['event_id']}: only {len(window)} rows "
              f"(need {SEQUENCE_LENGTH}). Will use whole performance dataset as fallback.")
        window = perf_df.copy()

    if len(window) < SEQUENCE_LENGTH:
        print(f"  [LSTM] Not enough rows even in full dataset ({len(window)}). Skipping.")
        return None

    # Scale features
    avail_cols = [c for c in FEATURE_COLUMNS if c in window.columns]
    window = _scale_features(window, avail_cols)

    # Pad missing feature columns with 0
    for col in FEATURE_COLUMNS:
        if col not in window.columns:
            window[col] = 0.0

    data = window[FEATURE_COLUMNS].values

    # Build sequences
    sequences = []
    for i in range(len(data) - SEQUENCE_LENGTH):
        sequences.append(data[i: i + SEQUENCE_LENGTH])

    if not sequences:
        return None

    X = np.array(sequences, dtype=np.float32)
    features = lstm_extractor.predict(X, verbose=0)  # shape (N, 16)
    mean_feat = features.mean(axis=0)                # shape (16,)
    print(f"  [LSTM] event {event['event_id']}: {len(sequences)} sequences "
          f"-> feature dim={mean_feat.shape[0]}")
    return mean_feat


def extract_cnn_features(
    event: dict,
    cnn_extractor: tf.keras.Model,
) -> np.ndarray | None:
    """
    Load the event's PNG image and run it through the CNN feature extractor.
    Returns a 128-dim feature vector.
    """
    img_path = os.path.join(IMAGE_FOLDER, event["image_file"])
    img = load_img(img_path, color_mode="grayscale",
                   target_size=(IMG_HEIGHT, IMG_WIDTH))
    arr = img_to_array(img) / 255.0               # (256, 256, 1)
    arr = np.expand_dims(arr, axis=0)              # (1, 256, 256, 1)
    feat = cnn_extractor.predict(arr, verbose=0)   # (1, 128)
    feat = feat.flatten()                          # (128,)
    print(f"  [CNN]  event {event['event_id']}: feature dim={feat.shape[0]}")
    return feat


# ---------------------------------------------------------
# STEP 4 — BUILD TRAINING DATASET WITH AUGMENTATION
# ---------------------------------------------------------
def build_fusion_dataset(
    events: list[dict],
    perf_df: pd.DataFrame,
    lstm_extractor: tf.keras.Model,
    cnn_extractor: tf.keras.Model,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (lstm_feat || cnn_feat) for each event, then augment with
    Gaussian noise to create enough samples for training.
    """
    X_real, y_real = [], []

    for event in events:
        lstm_feat = extract_lstm_features(event, perf_df, lstm_extractor)
        cnn_feat  = extract_cnn_features(event, cnn_extractor)

        if lstm_feat is None or cnn_feat is None:
            print(f"  WARNING: Skipping event {event['event_id']} "
                  "— could not extract features from both branches.")
            continue

        fused = np.concatenate([lstm_feat, cnn_feat])  # (16 + 128,) = (144,)
        X_real.append(fused)
        y_real.append(int(event["label"]))
        print(f"  Fused vector dim: {fused.shape[0]}  label={event['label']}")

    if len(X_real) == 0:
        raise ValueError(
            "No valid matched pairs could be extracted.\n"
            "Make sure event_registry.csv has rows with valid perf_start, "
            "perf_end, and image_file values pointing to existing PNG files."
        )

    X_real = np.array(X_real, dtype=np.float32)
    y_real = np.array(y_real, dtype=np.float32)

    unique_classes = np.unique(y_real)
    print(f"\n  Real samples: {len(X_real)} "
          f"| Classes: {dict(zip(*np.unique(y_real, return_counts=True)))}")

    if len(unique_classes) < 2:
        raise ValueError(
            f"Only class {unique_classes[0]} found in matched pairs.\n"
            "You need at least one normal (label=0) AND one suspicious (label=1) "
            "event in event_registry.csv with matching PNG images."
        )

    # Augment with Gaussian noise
    rng = np.random.default_rng(seed=42)
    X_aug, y_aug = [X_real], [y_real]
    for _ in range(AUGMENT_N):
        noise = rng.normal(0, NOISE_SIGMA, X_real.shape).astype(np.float32)
        X_aug.append(np.clip(X_real + noise, 0.0, 1.0))
        y_aug.append(y_real)

    X = np.concatenate(X_aug, axis=0)
    y = np.concatenate(y_aug, axis=0)
    print(f"  After augmentation: {len(X)} samples (x{AUGMENT_N+1} copies + noise)")
    return X, y


# ---------------------------------------------------------
# STEP 5 — FUSION HEAD MODEL
# ---------------------------------------------------------
def build_fusion_head(input_dim: int) -> tf.keras.Model:
    """
    Small dense classifier that takes the concatenated feature vector
    and outputs a normal/suspicious probability.
    """
    inp = layers.Input(shape=(input_dim,), name="fused_features")
    x = layers.Dense(64, activation="relu", name="fusion_dense1")(inp)
    x = layers.Dropout(0.3, name="fusion_dropout")(x)
    x = layers.Dense(32, activation="relu", name="fusion_dense2")(x)
    out = layers.Dense(1, activation="sigmoid", name="fusion_output")(x)

    model = models.Model(inputs=inp, outputs=out, name="HyperMD_FusionHead")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("=" * 66)
    print("  HyperMD-Enhanced -- Fusion Model Training (Module 4C)")
    print("=" * 66)

    # --- Load base models ---
    print("\n[1/6] Loading trained base models...")
    for path in [LSTM_MODEL_PATH, CNN_MODEL_PATH]:
        if not os.path.isfile(path):
            print(f"  ERROR: {path} not found.")
            print("  Run lstm_model.py and cnn_model.py first.")
            return

    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    print(f"  LSTM model loaded: {LSTM_MODEL_PATH}")
    cnn_model  = tf.keras.models.load_model(CNN_MODEL_PATH)
    print(f"  CNN model loaded : {CNN_MODEL_PATH}")

    # --- Build feature extractors ---
    print("\n[2/6] Building feature extractors...")
    lstm_extractor = build_lstm_extractor(lstm_model)
    cnn_extractor  = build_cnn_extractor(cnn_model)

    # Save the feature extractors (needed by inference.py and explainability/)
    os.makedirs(os.path.dirname(LSTM_FEAT_PATH), exist_ok=True)
    lstm_extractor.save(LSTM_FEAT_PATH)
    cnn_extractor.save(CNN_FEAT_PATH)
    print(f"  Saved lstm_features -> {os.path.abspath(LSTM_FEAT_PATH)}")
    print(f"  Saved cnn_features  -> {os.path.abspath(CNN_FEAT_PATH)}")

    # --- Load performance data ---
    print("\n[3/6] Loading performance data...")
    if not os.path.isfile(PERFORMANCE_FILE):
        print(f"  ERROR: {PERFORMANCE_FILE} not found.")
        return
    perf_df = pd.read_csv(PERFORMANCE_FILE)
    perf_df["timestamp"] = pd.to_datetime(perf_df["timestamp"], errors="coerce")
    perf_df = perf_df.dropna(subset=["timestamp"])
    print(f"  Loaded {len(perf_df)} performance rows.")

    # --- Load event registry ---
    print("\n[4/6] Loading matched event pairs from event_registry.csv...")
    events = load_event_registry()

    # --- Extract features and build dataset ---
    print("\n[5/6] Extracting features and building fusion dataset...")
    X, y = build_fusion_dataset(events, perf_df, lstm_extractor, cnn_extractor)
    input_dim = X.shape[1]
    print(f"\n  Final dataset shape: {X.shape}  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- Train fusion head ---
    print("\n[6/6] Training fusion head...")
    unique = np.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(unique) > 1 else None,
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    fusion_model = build_fusion_head(input_dim)
    fusion_model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
    ]

    history = fusion_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate ---
    y_pred_probs = fusion_model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["normal", "suspicious"],
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Save ---
    os.makedirs(os.path.dirname(FUSION_MODEL_PATH), exist_ok=True)
    fusion_model.save(FUSION_MODEL_PATH)
    print(f"\nFusion model saved -> {os.path.abspath(FUSION_MODEL_PATH)}")
    print("=" * 66)
    print("  Step 1 complete. Proceed to Step 2: models/inference.py")
    print("=" * 66)


if __name__ == "__main__":
    main()
