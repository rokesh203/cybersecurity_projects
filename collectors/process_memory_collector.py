"""
HyperMD-Enhanced - Module 2B: Detailed Process Memory Collector
Collects a deeper memory breakdown per process than Module 1's basic RSS.

Save at: D:\Rokesh Project\HyperMD-Enhanced\collectors\process_memory_collector.py
"""

import psutil
import csv
import os
import time
from datetime import datetime

SCAN_INTERVAL_SECONDS = 3
OUTPUT_FILE = "process_memory_details.csv"

CSV_HEADERS = [
    "timestamp",
    "pid",
    "process_name",
    "rss_mb",          # Resident Set Size - physical RAM actually used
    "vms_mb",          # Virtual Memory Size - total virtual address space
    "private_mb",      # Memory not shared with other processes
    "working_set_mb",  # Windows-specific: current working set
    "num_threads",
]


def init_csv_file(file_path, headers):
    if not os.path.isfile(file_path):
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def collect_process_memory():
    rows = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pid = proc.pid
            name = proc.info["name"]

            mem = proc.memory_info()
            rss_mb = round(mem.rss / (1024 * 1024), 2)
            vms_mb = round(mem.vms / (1024 * 1024), 2)

            # memory_full_info gives private bytes, but needs more
            # permission and is slower - handle gracefully.
            try:
                full_mem = proc.memory_full_info()
                private_mb = round(full_mem.private / (1024 * 1024), 2)
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                private_mb = None

            # working set is Windows-specific; psutil exposes it as
            # part of memory_info() on Windows under a different name
            # depending on version, so we fall back to rss if unavailable.
            working_set_mb = getattr(mem, "wset", None)
            working_set_mb = (
                round(working_set_mb / (1024 * 1024), 2)
                if working_set_mb is not None
                else rss_mb
            )

            try:
                num_threads = proc.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                num_threads = None

            rows.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pid,
                name,
                rss_mb,
                vms_mb,
                private_mb,
                working_set_mb,
                num_threads,
            ])

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return rows


def main():
    print("=" * 60)
    print("HyperMD-Enhanced - Process Memory Collector (Module 2B)")
    print("=" * 60)
    print(f"Output -> {OUTPUT_FILE}")
    print("Press Ctrl + C to stop.\n")

    init_csv_file(OUTPUT_FILE, CSV_HEADERS)
    scan_count = 0

    try:
        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            while True:
                rows = collect_process_memory()
                writer.writerows(rows)
                f.flush()
                scan_count += 1
                print(f"Scan #{scan_count}: {len(rows)} processes recorded.")
                time.sleep(SCAN_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nStopped. Data saved to {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()