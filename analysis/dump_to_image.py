"""
HyperMD-Enhanced - Analysis Module: Dump-to-Image Converter
============================================================
Converts a Windows memory dump (.dmp) file into a fixed-width
grayscale PNG image using the standard malware-visualization
technique (Nataraj et al., 2011).

How it works
------------
1. Read the raw bytes of the dump file.
2. Treat each byte as a grayscale pixel (0-255).
3. Reshape into rows of WIDTH pixels (padding the last row if needed).
4. Save as a PNG image.

The resulting image encodes the full memory layout visually.
Malicious and benign processes tend to produce recognizably
different texture patterns, which a CNN can learn to distinguish.

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\analysis\\dump_to_image.py

Usage
-----
# Convert a single dump file, label 0 (normal)
    python analysis/dump_to_image.py --dump memory_dumps/proc_1234.dmp --label 0

# Convert with a custom output filename
    python analysis/dump_to_image.py --dump memory_dumps/proc_1234.dmp --label 1 --out stress_session_01.png

# Import and call directly from another script
    from analysis.dump_to_image import convert_dump_to_image
    img_path = convert_dump_to_image("memory_dumps/proc.dmp", label=1)
"""

import argparse
import csv
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
IMAGE_WIDTH = 256          # standard width used in malware viz literature
IMAGE_FOLDER = "memory_images"
LABEL_CSV = os.path.join(IMAGE_FOLDER, "image_labels.csv")

LABEL_CSV_HEADERS = [
    "event_id",
    "filename",
    "label",
    "perf_start",
    "perf_end",
    "notes",
]


# ---------------------------------------------------------
# CORE CONVERSION
# ---------------------------------------------------------
def _read_dump_bytes(dump_path: str) -> bytes:
    """Read all bytes from a .dmp file, stripping a Windows minidump
    header if present so we get the raw memory content."""
    with open(dump_path, "rb") as f:
        data = f.read()

    # Windows minidumps start with "MDMP" magic (4 bytes).
    # Skip the first 4 KB of header so the image shows actual memory,
    # not metadata.  If the file is smaller than 4 KB, use it as-is.
    HEADER_SKIP = 4096
    if len(data) > HEADER_SKIP and data[:4] == b"MDMP":
        data = data[HEADER_SKIP:]

    return data


def _bytes_to_image_array(data: bytes, width: int) -> np.ndarray:
    """Reshape a flat byte sequence into a 2D uint8 array of the
    given width, padding the last row with zeros if needed."""
    arr = np.frombuffer(data, dtype=np.uint8)

    # Pad to a multiple of width
    remainder = len(arr) % width
    if remainder != 0:
        pad = width - remainder
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])

    num_rows = len(arr) // width
    return arr.reshape((num_rows, width))


def convert_dump_to_image(
    dump_path: str,
    label: int = 0,
    event_id: str = "",
    perf_start: str = "",
    perf_end: str = "",
    notes: str = "",
    output_filename: str = "",
) -> str:
    """
    Convert a .dmp file to a grayscale PNG and register it in
    image_labels.csv.

    Parameters
    ----------
    dump_path      : path to the .dmp file
    label          : 0 = normal, 1 = suspicious
    event_id       : shared ID linking this image to a perf window
    perf_start     : ISO timestamp of performance window start
    perf_end       : ISO timestamp of performance window end
    notes          : optional human note (e.g. "benign stress test run 2")
    output_filename: override the auto-generated PNG filename

    Returns
    -------
    Absolute path of the saved PNG file.
    """
    dump_path = os.path.abspath(dump_path)
    if not os.path.isfile(dump_path):
        raise FileNotFoundError(f"Dump file not found: {dump_path}")

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    # Auto-generate output filename from dump name if not supplied
    if not output_filename:
        stem = Path(dump_path).stem  # e.g. "process_1234_20260917_215847"
        label_tag = "normal" if label == 0 else "suspicious"
        if event_id:
            output_filename = f"{event_id}_{label_tag}.png"
        else:
            output_filename = f"{stem}_{label_tag}.png"

    output_path = os.path.join(IMAGE_FOLDER, output_filename)

    print(f"  Reading dump: {dump_path}")
    print(f"  File size   : {os.path.getsize(dump_path) / (1024*1024):.1f} MB")

    data = _read_dump_bytes(dump_path)
    img_array = _bytes_to_image_array(data, IMAGE_WIDTH)

    # Resize to a fixed height (256) so all images are the same shape
    # (required for CNN batching). We use NEAREST to preserve byte values.
    TARGET_HEIGHT = 256
    img = Image.fromarray(img_array, mode="L")  # "L" = 8-bit grayscale
    img = img.resize((IMAGE_WIDTH, TARGET_HEIGHT), Image.NEAREST)
    img.save(output_path)

    print(f"  Saved image : {os.path.abspath(output_path)}")
    print(f"  Image size  : {IMAGE_WIDTH}x{TARGET_HEIGHT} px")

    # Register in image_labels.csv
    _register_label(
        filename=output_filename,
        label=label,
        event_id=event_id,
        perf_start=perf_start,
        perf_end=perf_end,
        notes=notes,
    )

    return os.path.abspath(output_path)


def _register_label(
    filename: str,
    label: int,
    event_id: str = "",
    perf_start: str = "",
    perf_end: str = "",
    notes: str = "",
) -> None:
    """Write or append a row to image_labels.csv."""
    file_exists = os.path.isfile(LABEL_CSV)

    with open(LABEL_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(LABEL_CSV_HEADERS)
        writer.writerow([event_id, filename, label, perf_start, perf_end, notes])

    print(f"  Registered  : {filename} -> label={label} in {LABEL_CSV}")


# ---------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a .dmp file to a grayscale PNG memory image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dump", required=True, help="Path to the .dmp file.")
    p.add_argument(
        "--label", type=int, choices=[0, 1], required=True,
        help="0 = normal/benign, 1 = suspicious."
    )
    p.add_argument("--event-id", default="", help="Optional event_id tag.")
    p.add_argument("--perf-start", default="", help="Performance window start timestamp.")
    p.add_argument("--perf-end", default="", help="Performance window end timestamp.")
    p.add_argument("--notes", default="", help="Free-text note about this sample.")
    p.add_argument("--out", default="", help="Override output PNG filename.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = convert_dump_to_image(
        dump_path=args.dump,
        label=args.label,
        event_id=args.event_id,
        perf_start=args.perf_start,
        perf_end=args.perf_end,
        notes=args.notes,
        output_filename=args.out,
    )
    print(f"\nDone. Image saved to: {result}")

