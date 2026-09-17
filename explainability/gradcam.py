"""
HyperMD-Enhanced - Module 5B: Grad-CAM Explainer (CNN / Image Branch)
======================================================================
Computes a Gradient-weighted Class Activation Map (Grad-CAM) heatmap
for the CNN branch of the fusion model.

Grad-CAM highlights which spatial regions of the memory image were
most influential in the CNN's classification decision.

What it tells you
-----------------
"The CNN flagged this sample because of unusual texture in the
memory region at rows 64-128 (upper-right quadrant)."

Output
------
A dict with:
  "heatmap"      : np.ndarray shape (256, 256) float32, values in [0, 1]
  "overlay"      : np.ndarray shape (256, 256, 3) uint8  (heatmap on image)
  "hotspot_desc" : human-readable description of the hottest region
  "text_summary" : one-line explanation string

The heatmap and overlay can be saved as PNGs by the dashboard.

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\explainability\\gradcam.py

Usage
-----
    from explainability.gradcam import explain_cnn
    from models.inference import load_models

    ctx = load_models()
    result = explain_cnn(image_path, ctx)
    print(result["text_summary"])

    # Save heatmap overlay
    from PIL import Image
    Image.fromarray(result["overlay"]).save("heatmap_overlay.png")
"""

import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

IMG_HEIGHT = 256
IMG_WIDTH  = 256

# The last Conv2D layer in our CNN — named in cnn_model.py
GRADCAM_LAYER_NAME = "conv_block3_conv"


def _load_image(image_path: str) -> np.ndarray:
    """Load and normalise a grayscale PNG -> (1, 256, 256, 1) float32."""
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path, color_mode="grayscale",
                   target_size=(IMG_HEIGHT, IMG_WIDTH))
    arr = img_to_array(img) / 255.0
    return arr[np.newaxis, :, :, :].astype(np.float32)


def _compute_gradcam(cnn_extractor_model, image_arr: np.ndarray) -> np.ndarray:
    """
    Compute Grad-CAM heatmap using the last Conv2D layer of the CNN.

    We use the CNN feature extractor (not the full CNN) and backpropagate
    through the sum of the feature vector output w.r.t. the target conv layer.

    Returns a (256, 256) float32 heatmap in [0, 1].
    """
    import tensorflow as tf

    # Build a sub-model from image input to the Grad-CAM target conv layer
    try:
        conv_layer = cnn_extractor_model.get_layer(GRADCAM_LAYER_NAME)
    except ValueError:
        # If the exact layer name is not found, fall back to the last Conv2D
        conv_layer = None
        for layer in reversed(cnn_extractor_model.layers):
            if "conv" in layer.name.lower():
                conv_layer = layer
                break

    if conv_layer is None:
        # Absolute fallback: return uniform heatmap
        return np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32) * 0.5

    # Warm up to ensure layer output tensor is built
    cnn_extractor_model(image_arr, training=False)

    grad_model = tf.keras.Model(
        inputs=cnn_extractor_model.inputs,
        outputs=[conv_layer.output, cnn_extractor_model.output],
    )

    img_tensor = tf.constant(image_arr, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, feature_vec = grad_model(img_tensor, training=False)
        # Score = sum of feature vector (proxy for "how suspicious")
        score = tf.reduce_sum(feature_vec)

    # Gradients of score w.r.t. convolutional layer outputs
    grads = tape.gradient(score, conv_outputs)     # (1, H, W, C)

    # Pool gradients over spatial dimensions -> channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight the conv output channels
    conv_out = conv_outputs[0]                            # (H, W, C)
    for ch in range(conv_out.shape[-1]):
        conv_out = conv_out.numpy() if not isinstance(conv_out, np.ndarray) else conv_out
        conv_out[:, :, ch] *= pooled_grads[ch].numpy()

    heatmap = np.mean(conv_out, axis=-1)                  # (H, W)
    heatmap = np.maximum(heatmap, 0)                      # ReLU

    # Normalise to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Resize to original image size (256x256) using simple repeat
    from PIL import Image as PILImage
    heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((IMG_WIDTH, IMG_HEIGHT), PILImage.BILINEAR)
    heatmap_full = np.array(heatmap_pil) / 255.0          # (256, 256) float32

    return heatmap_full.astype(np.float32)


def _heatmap_to_overlay(image_arr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Blend the grayscale memory image with the Grad-CAM heatmap.

    image_arr : (1, 256, 256, 1) float32
    heatmap   : (256, 256) float32 [0,1]
    Returns   : (256, 256, 3) uint8 RGB overlay
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Grayscale image -> RGB
    img_gray = image_arr[0, :, :, 0]               # (256, 256) float32
    img_rgb  = np.stack([img_gray]*3, axis=-1)      # (256, 256, 3)

    # Colormap for heatmap (red = hot)
    colormap  = cm.jet(heatmap)[:, :, :3]           # (256, 256, 3) float32
    overlay   = 0.5 * img_rgb + 0.5 * colormap     # blend
    overlay   = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def _describe_hotspot(heatmap: np.ndarray) -> str:
    """
    Find the hottest 64x64 region and return a human-readable label.
    """
    # Slide a 64x64 window, find the max mean activation
    win = 64
    best_val   = -1.0
    best_r, best_c = 0, 0
    for r in range(0, IMG_HEIGHT - win + 1, win):
        for c in range(0, IMG_WIDTH - win + 1, win):
            val = heatmap[r:r+win, c:c+win].mean()
            if val > best_val:
                best_val   = val
                best_r, best_c = r, c

    # Map pixel coords to memory region names
    row_label = "upper" if best_r < IMG_HEIGHT // 2 else "lower"
    col_label = "left"  if best_c < IMG_WIDTH  // 2 else "right"
    region    = f"{row_label}-{col_label} quadrant"
    byte_start = best_r * IMG_WIDTH + best_c          # approx byte offset

    return (
        f"Hottest region: {region} (pixel rows {best_r}-{best_r+win}, "
        f"cols {best_c}-{best_c+win}) — "
        f"approx. memory offset 0x{byte_start * 256:08X}"
    )


def explain_cnn(
    image_path: str,
    ctx: dict,
    save_overlay_path: str = "",
) -> dict:
    """
    Compute a Grad-CAM explanation for one memory image.

    Parameters
    ----------
    image_path        : path to the 256x256 grayscale PNG.
    ctx               : model context from models.inference.load_models().
    save_overlay_path : if non-empty, save the overlay PNG to this path.

    Returns
    -------
    dict with keys:
        heatmap      (ndarray 256x256 float32)
        overlay      (ndarray 256x256x3 uint8)
        hotspot_desc (str)
        text_summary (str)
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    cnn_ext = ctx["cnn_extractor"]

    image_arr = _load_image(image_path)
    heatmap   = _compute_gradcam(cnn_ext, image_arr)
    overlay   = _heatmap_to_overlay(image_arr, heatmap)
    hotspot   = _describe_hotspot(heatmap)

    max_act   = float(heatmap.max())
    text_summary = (
        f"CNN attention: {hotspot}. "
        f"Max activation intensity: {max_act:.3f}."
    )

    if save_overlay_path:
        from PIL import Image as PILImage
        PILImage.fromarray(overlay).save(save_overlay_path)

    return {
        "heatmap":      heatmap,
        "overlay":      overlay,
        "hotspot_desc": hotspot,
        "text_summary": text_summary,
    }

