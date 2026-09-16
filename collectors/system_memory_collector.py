"""
HyperMD-Enhanced - Module 2A: System Physical Memory Collector
Collects detailed system-wide memory statistics (not per-process).

Save at: D:\Rokesh Project\HyperMD-Enhanced\collectors\system_memory_collector.py
"""

import psutil
import csv
import os
import time
from datetime import datetime

SCAN_INTERVAL_SECONDS = 2
OUTPUT_FILE = "system_memory_data.csv"

CSV_HEADERS = [
    "timestamp",
    "total_mb",
    "available_mb",
    "used_mb",
    "free_mb",
    "percent_used",
    "swap_total_mb",
    "swap_used_mb",
    "swap_percent",
]


def init_csv_file(file_path, headers):
    if not os.path.isfile(file_path):
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


def collect_system_memory():
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        round(vm.total / (1024 * 1024), 2),
        round(vm.available / (1024 * 1024), 2),
        round(vm.used / (1024 * 1024), 2),
        round(vm.free / (1024 * 1024), 2),
        vm.percent,
        round(sm.total / (1024 * 1024), 2),
        round(sm.used / (1024 * 1024), 2),
        sm.percent,
    ]


def main():
    print("=" * 60)
    print("HyperMD-Enhanced - System Memory Collector (Module 2A)")
    print("=" * 60)
    print(f"Output -> {OUTPUT_FILE}")
    print("Press Ctrl + C to stop.\n")

    init_csv_file(OUTPUT_FILE, CSV_HEADERS)
    count = 0

    try:
        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            while True:
                row = collect_system_memory()
                writer.writerow(row)
                f.flush()
                count += 1
                print(f"Scan #{count}: {row[5]}% used | Available: {row[2]} MB")
                time.sleep(SCAN_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nStopped. {count} records saved to {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()