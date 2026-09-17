"""
HyperMD-Enhanced - Module 4B: CNN Model for Memory Dump Images
===============================================================
Trains a Convolutional Neural Network to classify grayscale memory
images (256x256 px) as normal (0) or suspicious (1).

The images are produced by analysis/dump_to_image.py from real
Windows .dmp files. This is the visual/spatial branch of the
multi-modal detection pipeline.

Architecture
------------
Input: 256x256x1 grayscale image
  -> Conv2D(32) + MaxPool
  -> Conv2D(64) + MaxPool
  -> Conv2D(128) + MaxPool          ← Grad-CAM targets this layer
  -> GlobalAveragePooling2D
  -> Dense(128, relu)               ← feature vector extracted here for fusion
  -> Dropout(0.4)
  -> Dense(1, sigmoid)              ← final normal/suspicious output

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\models\\cnn_model.py

Run
---
    .venv\\Scripts\\python.exe models/cnn_model.py

Expected output
---------------
  - Training accuracy and loss for 20 epochs
  - Classification report (precision, recall, F1)
  - Saved model: models/saved_models/cnn_model.h5
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF INFO/WARNING noise

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
IMAGE_FOLDER = "memory_images"
LABEL_CSV    = os.path.join(IMAGE_FOLDER, "image_labels.csv")
MODEL_OUTPUT = "models/saved_models/cnn_model.h5"

IMG_HEIGHT = 256
IMG_WIDTH  = 256
CHANNELS   = 1    # grayscale

EPOCHS     = 20
BATCH_SIZE = 8    # small because memory images are large and we may have few samples
TEST_SPLIT = 0.2
RANDOM_SEED = 42


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
def load_images_and_labels() -> tuple[np.ndarray, np.ndarray]:
    """
    Read image_labels.csv, load each PNG as a grayscale array,
    and return (X, y) arrays ready for CNN training.
    """
    if not os.path.isfile(LABEL_CSV):
        raise FileNotFoundError(
            f"{LABEL_CSV} not found. Run event_session.py first to generate labeled images."
        )

    df = pd.read_csv(LABEL_CSV)

    # Support both old schema (filename,label) and new schema (event_id,filename,label,...)
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"image_labels.csv must have at least 'filename' and 'label' columns. "
            f"Found: {list(df.columns)}"
        )

    images, labels = [], []
    skipped = 0

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGE_FOLDER, row["filename"])
        if not os.path.isfile(img_path):
            print(f"  WARNING: image not found, skipping: {img_path}")
            skipped += 1
            continue

        img = load_img(img_path, color_mode="grayscale",
                       target_size=(IMG_HEIGHT, IMG_WIDTH))
        arr = img_to_array(img) / 255.0   # normalise to [0, 1]
        images.append(arr)
        labels.append(int(row["label"]))

    if skipped > 0:
        print(f"  Skipped {skipped} missing image(s).")

    if len(images) == 0:
        raise ValueError(
            "No valid images found. Run event_session.py to generate labeled images first."
        )

    X = np.array(images)   # shape: (N, 256, 256, 1)
    y = np.array(labels)   # shape: (N,)

    print(f"  Loaded {len(X)} images.")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        label_name = "normal" if cls == 0 else "suspicious"
        print(f"  Class {cls} ({label_name}): {cnt} samples")

    return X, y


# ---------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------
def build_cnn(input_shape: tuple) -> tf.keras.Model:
    """
    Build the CNN model.

    The third Conv2D block is named 'conv_block3_conv' so that
    Grad-CAM in the explainability module can reliably target it
    by name without hard-coding an integer layer index.
    """
    inp = layers.Input(shape=input_shape, name="image_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_block1_conv")(inp)
    x = layers.MaxPooling2D((2, 2), name="conv_block1_pool")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_block2_conv")(x)
    x = layers.MaxPooling2D((2, 2), name="conv_block2_pool")(x)

    # Block 3 -- Grad-CAM target layer
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_block3_conv")(x)
    x = layers.MaxPooling2D((2, 2), name="conv_block3_pool")(x)

    # Global pooling -> feature vector
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense feature layer (extracted for fusion)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout")(x)

    # Output
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="HyperMD_CNN")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------
# TRAINING HELPERS
# ---------------------------------------------------------
def _augment_with_flips(X: np.ndarray, y: np.ndarray):
    """
    Simple data augmentation: add horizontally and vertically
    flipped copies of each image. Doubles/quadruples the dataset,
    which is crucial when we have very few real samples.
    """
    X_hflip = X[:, :, ::-1, :]        # horizontal flip
    X_vflip = X[:, ::-1, :, :]        # vertical flip
    X_aug = np.concatenate([X, X_hflip, X_vflip], axis=0)
    y_aug = np.concatenate([y, y, y], axis=0)
    print(f"  After augmentation: {len(X_aug)} samples.")
    return X_aug, y_aug


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("=" * 62)
    print("  HyperMD-Enhanced -- CNN Model Training (Module 4B)")
    print("=" * 62)

    print("\n[1/5] Loading images...")
    X, y = load_images_and_labels()

    # Require both classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(
            "\nERROR: Only one class found in image_labels.csv.\n"
            "You need at least one normal (label=0) AND one suspicious (label=1) image.\n"
            "Run event_session.py twice -- once without the stress test (label=0)\n"
            "and once with the stress test (label=1)."
        )
        return

    print("\n[2/5] Augmenting data...")
    X, y = _augment_with_flips(X, y)

    print("\n[3/5] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[4/5] Building and training CNN...")
    model = build_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[5/5] Evaluating and saving...")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["normal", "suspicious"],
                                zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    model.save(MODEL_OUTPUT)
    print(f"\nModel saved -> {os.path.abspath(MODEL_OUTPUT)}")
    print("=" * 62)


if __name__ == "__main__":
    main()

