"""
HyperMD-Enhanced - Step 0: Event Session Orchestrator
======================================================
Single command to run one complete, fully-labeled data collection
session. Replaces all manual steps (copy-pasting PIDs, editing CSVs,
running ProcDump separately, running dump_to_image separately).

What it does (in order)
-----------------------
1. Generates a unique event_id (e.g. "evt_20260917_221500_normal")
2. Starts logging the start timestamp in performance_data.csv
3. Runs either:
     --label 0  -> 60-second idle period (normal baseline)
     --label 1  -> benign_stress_test.py (suspicious/high-resource)
4. Records the end timestamp
5. Dumps the Python process with ProcDump (fully automatic, no PID entry)
6. Converts the .dmp -> grayscale PNG via analysis/dump_to_image.py
7. Tags all performance_data.csv rows in the session window with
   the event_id and label
8. Appends a row to event_registry.csv (the master pairing table)
9. Appends a row to memory_images/image_labels.csv

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\collectors\\event_session.py

Usage
-----
# Collect a NORMAL baseline session (60 seconds, no stress)
    .venv\\Scripts\\python.exe collectors/event_session.py --label 0

# Collect a SUSPICIOUS session (stress test, 60 seconds)
    .venv\\Scripts\\python.exe collectors/event_session.py --label 1

# Custom duration
    .venv\\Scripts\\python.exe collectors/event_session.py --label 1 --duration 120

# Add a note
    .venv\\Scripts\\python.exe collectors/event_session.py --label 0 --notes "idle desktop, no browser"

Prerequisites
-------------
- ProcDump installed at C:\\Tools\\ProcDump\\procdump.exe
- Virtual environment activated (or use full .venv path as above)
- Run from the project root: D:\\Rokesh Project\\HyperMD-Enhanced\\
"""

import argparse
import csv
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path so sibling imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.dump_to_image import convert_dump_to_image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PROCDUMP_PATH      = r"C:\Tools\ProcDump\procdump.exe"
DUMP_FOLDER        = "memory_dumps"
PERFORMANCE_FILE   = "performance_data.csv"
EVENT_REGISTRY     = "event_registry.csv"
LABEL_NAMES        = {0: "normal", 1: "suspicious"}

EVENT_REGISTRY_HEADERS = [
    "event_id",
    "label",
    "label_name",
    "perf_start",
    "perf_end",
    "dump_file",
    "image_file",
    "notes",
]

PERF_CSV_HEADERS = [
    "timestamp", "cpu_percent", "memory_percent", "process_count",
    "disk_read_bytes", "disk_write_bytes",
]


# ---------------------------------------------------------
# PERFORMANCE COLLECTION (background thread)
# ---------------------------------------------------------
def _collect_performance(stop_event: threading.Event) -> None:
    """Background thread: collect system perf every second into performance_data.csv."""
    try:
        import psutil
    except ImportError:
        print("  [perf] psutil not available -- skipping performance collection.")
        return

    file_exists = os.path.isfile(PERFORMANCE_FILE)
    with open(PERFORMANCE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(PERF_CSV_HEADERS)

        while not stop_event.is_set():
            try:
                mem = psutil.virtual_memory()
                disk = psutil.disk_io_counters()
                row = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    round(psutil.cpu_percent(interval=None), 2),
                    round(mem.percent, 2),
                    len(psutil.pids()),
                    disk.read_bytes if disk else 0,
                    disk.write_bytes if disk else 0,
                ]
                writer.writerow(row)
                f.flush()
            except Exception:
                pass
            time.sleep(1.0)


# ---------------------------------------------------------
# STRESS TEST RUNNER (in-process, timed mode)
# ---------------------------------------------------------
def _cpu_burn_worker(duration: float) -> None:
    """CPU-burn worker process (copied here so multiprocessing works cleanly)."""
    end = time.monotonic() + duration
    x = 0.0
    while time.monotonic() < end:
        x = (x + 1.23456) * 0.99999


def _run_stress_test(duration: float, memory_mb: int, num_workers: int) -> None:
    """Launch CPU workers + hold memory for `duration` seconds, then clean up."""
    # Start CPU workers
    workers = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=_cpu_burn_worker,
            args=(duration + 5,),
            daemon=True,
            name=f"cpu-worker-{i}",
        )
        p.start()
        workers.append(p)
        print(f"  [stress] CPU worker {i} started (PID {p.pid})")

    # Allocate and hold memory
    print(f"  [stress] Allocating {memory_mb} MB of memory...")
    try:
        block = bytearray(memory_mb * 1024 * 1024)
        for i in range(0, len(block), 4096):
            block[i] = 0xAB
        print(f"  [stress] Memory held. Stress running for {duration}s...")
        time.sleep(duration)
    except MemoryError:
        print(f"  [stress] WARNING: Could not allocate {memory_mb} MB. Continuing anyway.")
        time.sleep(duration)
    finally:
        if 'block' in dir():
            del block

    # Terminate workers
    for p in workers:
        p.terminate()
    for p in workers:
        p.join(timeout=3)
    print("  [stress] All CPU workers stopped.")


# ---------------------------------------------------------
# PROCDUMP WRAPPER
# ---------------------------------------------------------
def _dump_this_process(event_id: str) -> str | None:
    """
    Dump the CURRENT Python process using ProcDump.
    Returns the path to the created .dmp file, or None on failure.
    """
    if not os.path.isfile(PROCDUMP_PATH):
        print(f"  [dump] ERROR: ProcDump not found at {PROCDUMP_PATH}")
        print("         Install ProcDump and ensure the path is correct.")
        return None

    os.makedirs(DUMP_FOLDER, exist_ok=True)
    pid = os.getpid()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_filename = f"{event_id}_{timestamp}.dmp"
    dump_path = os.path.join(DUMP_FOLDER, dump_filename)

    print(f"  [dump] Dumping PID {pid} -> {dump_path}")
    result = subprocess.run(
        [PROCDUMP_PATH, "-ma", "-accepteula", str(pid), dump_path],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"  [dump] {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"  [dump] {line}")

    # ProcDump may add ".dmp" suffix again or create name differently
    # Search for the actual file
    actual_path = dump_path
    if not os.path.isfile(actual_path):
        # Try with doubled extension (.dmp.dmp)
        alt = dump_path + ".dmp"
        if os.path.isfile(alt):
            actual_path = alt
        else:
            # Search dump folder for any file containing the event_id
            for f in os.listdir(DUMP_FOLDER):
                if event_id in f and f.endswith(".dmp"):
                    actual_path = os.path.join(DUMP_FOLDER, f)
                    break

    if os.path.isfile(actual_path) and os.path.getsize(actual_path) > 0:
        size_mb = os.path.getsize(actual_path) / (1024 * 1024)
        print(f"  [dump] Dump complete. Size: {size_mb:.1f} MB")
        return actual_path
    else:
        print("  [dump] ERROR: Dump file was not created. Check ProcDump output above.")
        return None


# ---------------------------------------------------------
# PERFORMANCE TAGGING
# ---------------------------------------------------------
def _tag_performance_rows(event_id: str, label: int,
                           perf_start: str, perf_end: str) -> int:
    """
    Add event_id and label columns to performance_data.csv rows
    that fall within [perf_start - 30s, perf_end + 30s].
    The 30-second buffer handles any clock drift between the recording
    of perf_start/perf_end and the background thread's write timestamps.
    Returns the number of rows tagged.
    """
    if not os.path.isfile(PERFORMANCE_FILE):
        print(f"  [tag] WARNING: {PERFORMANCE_FILE} not found, skipping.")
        return 0

    try:
        import pandas as pd
        df = pd.read_csv(PERFORMANCE_FILE)
        if df.empty:
            print("  [tag] WARNING: performance_data.csv is empty.")
            return 0

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            print("  [tag] WARNING: No valid timestamps in performance_data.csv.")
            return 0

        # Add buffer of 30 seconds on each side to handle timing drift
        BUFFER_SECONDS = 30
        t_start = pd.to_datetime(perf_start) - pd.Timedelta(seconds=BUFFER_SECONDS)
        t_end   = pd.to_datetime(perf_end)   + pd.Timedelta(seconds=BUFFER_SECONDS)

        # Debug: show what's in the file vs what we're looking for
        ts_min = df["timestamp"].min()
        ts_max = df["timestamp"].max()
        print(f"  [tag] CSV timestamp range : {ts_min} -> {ts_max}")
        print(f"  [tag] Session window      : {t_start} -> {t_end}")

        mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)

        # Ensure columns exist before writing
        if "event_id" not in df.columns:
            df.insert(len(df.columns), "event_id", "")
        if "label" not in df.columns:
            df.insert(len(df.columns), "label", -1)

        df.loc[mask, "event_id"] = event_id
        df.loc[mask, "label"] = label

        # Write back -- convert timestamps to string first
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(PERFORMANCE_FILE, index=False)

        tagged = int(mask.sum())
        print(f"  [tag] Tagged {tagged} performance rows with event_id={event_id}, label={label}")
        return tagged

    except Exception as e:
        print(f"  [tag] WARNING: Could not tag performance rows: {e}")
        return 0



# ---------------------------------------------------------
# EVENT REGISTRY
# ---------------------------------------------------------
def _append_event_registry(
    event_id: str, label: int, perf_start: str, perf_end: str,
    dump_file: str, image_file: str, notes: str
) -> None:
    """Append a row to event_registry.csv."""
    file_exists = os.path.isfile(EVENT_REGISTRY)
    with open(EVENT_REGISTRY, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(EVENT_REGISTRY_HEADERS)
        writer.writerow([
            event_id,
            label,
            LABEL_NAMES.get(label, "unknown"),
            perf_start,
            perf_end,
            dump_file,
            image_file,
            notes,
        ])
    print(f"  [registry] Appended to {EVENT_REGISTRY}")


# ---------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------
def run_session(
    label: int,
    duration: float,
    memory_mb: int,
    notes: str,
) -> None:
    label_name = LABEL_NAMES.get(label, "unknown")
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    event_id = f"evt_{timestamp_tag}_{label_name}"

    print("=" * 66)
    print(f"  HyperMD-Enhanced -- Event Session Orchestrator")
    print("=" * 66)
    print(f"  event_id  : {event_id}")
    print(f"  label     : {label} ({label_name})")
    print(f"  duration  : {duration}s")
    print(f"  memory    : {memory_mb} MB (stress only)")
    if notes:
        print(f"  notes     : {notes}")
    print()

    # -- Phase 1: Start performance collector -----------------------------
    perf_stop = threading.Event()
    perf_thread = threading.Thread(
        target=_collect_performance,
        args=(perf_stop,),
        daemon=True,
        name="perf-collector",
    )
    # Record window start BEFORE sleep so all rows the thread writes are captured
    perf_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    perf_thread.start()
    # Let the collector prime its CPU baseline (one dummy measurement)
    time.sleep(2)
    print(f"  [phase 1] Performance collection started. Window start: {perf_start}")

    # -- Phase 2: Run the test scenario -----------------------------------
    print(f"\n  [phase 2] Running {'STRESS TEST' if label == 1 else 'IDLE BASELINE'} ({duration}s)...")
    if label == 1:
        num_workers = max(2, multiprocessing.cpu_count() // 2)
        _run_stress_test(
            duration=duration,
            memory_mb=memory_mb,
            num_workers=num_workers,
        )
    else:
        # Normal/baseline: just wait
        for remaining in range(int(duration), 0, -5):
            print(f"  [phase 2] Idle baseline -- {remaining}s remaining...", end="\r")
            time.sleep(min(5, remaining))
        print()

    perf_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [phase 2] Done. Window end: {perf_end}")

    # -- Phase 3: Memory dump ----------------------------------------------
    print(f"\n  [phase 3] Dumping this process with ProcDump...")
    dump_path = _dump_this_process(event_id)

    # -- Phase 4: Stop performance collector -------------------------------
    perf_stop.set()
    perf_thread.join(timeout=3)
    print(f"  [phase 4] Performance collector stopped.")

    if dump_path is None:
        print("\n  WARNING: No dump was created. Skipping image conversion.")
        print("  Session data (performance rows) will still be tagged.")
        dump_file_name = ""
        image_file_name = ""
    else:
        # -- Phase 5: Convert dump -> PNG ------------------------------------
        print(f"\n  [phase 5] Converting dump to grayscale PNG...")
        img_path = convert_dump_to_image(
            dump_path=dump_path,
            label=label,
            event_id=event_id,
            perf_start=perf_start,
            perf_end=perf_end,
            notes=notes,
        )
        dump_file_name  = os.path.basename(dump_path)
        image_file_name = os.path.basename(img_path)

    # -- Phase 6: Tag performance rows --------------------------------------
    print(f"\n  [phase 6] Tagging performance rows...")
    _tag_performance_rows(event_id, label, perf_start, perf_end)

    # -- Phase 7: Register event --------------------------------------------
    print(f"\n  [phase 7] Writing to event_registry.csv...")
    _append_event_registry(
        event_id=event_id,
        label=label,
        perf_start=perf_start,
        perf_end=perf_end,
        dump_file=dump_file_name,
        image_file=image_file_name,
        notes=notes,
    )

    # -- Summary ------------------------------------------------------------
    print()
    print("=" * 66)
    print("  SESSION COMPLETE")
    print("=" * 66)
    print(f"  event_id   : {event_id}")
    print(f"  label      : {label} ({label_name})")
    print(f"  perf window: {perf_start}  ->  {perf_end}")
    print(f"  dump file  : {dump_file_name or 'SKIPPED'}")
    print(f"  image file : {image_file_name or 'SKIPPED'}")
    print(f"  registry   : {os.path.abspath(EVENT_REGISTRY)}")
    print()
    print("  Next steps:")
    print("  * Run this again with --label 0 for a NORMAL session (if not done yet)")
    print("  * Run this again with --label 1 for a SUSPICIOUS session (if not done yet)")
    print("  * Once you have >=2 sessions (one of each class), train the CNN:")
    print("    .venv\\Scripts\\python.exe models/cnn_model.py")
    print("=" * 66)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "HyperMD-Enhanced Event Session Orchestrator.\n"
            "Runs a labeled data collection session end-to-end:\n"
            "  performance collection -> test scenario -> ProcDump -> PNG -> tagging."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--label", "-l",
        type=int, choices=[0, 1], required=True,
        help="0 = normal/benign baseline.  1 = suspicious (runs stress test).",
    )
    p.add_argument(
        "--duration", "-d",
        type=float, default=60.0, metavar="SECONDS",
        help="How long to run the scenario (default: 60s).",
    )
    p.add_argument(
        "--memory", "-m",
        type=int, default=200, metavar="MB",
        help="MB of RAM for the stress test (label=1 only, default: 200).",
    )
    p.add_argument(
        "--notes",
        type=str, default="",
        help="Optional free-text note stored in the event registry.",
    )
    return p.parse_args()


if __name__ == "__main__":
    multiprocessing.freeze_support()   # Required on Windows
    args = _parse_args()
    run_session(
        label=args.label,
        duration=args.duration,
        memory_mb=args.memory,
        notes=args.notes,
    )
