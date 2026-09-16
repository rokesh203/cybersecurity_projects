"""
HyperMD-Enhanced - Module 1: Improved Process-Level Collector
Collects detailed per-process information and tracks process
creation/termination events. Windows-compatible.

Save at: D:\Rokesh Project\HyperMD-Enhanced\collectors\process_collector.py
"""

import psutil
import pandas as pd
import time
import csv
import os
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SCAN_INTERVAL_SECONDS = 2
PROCESS_DATA_FILE = "process_data.csv"
EVENT_LOG_FILE = "process_events.csv"

# ---------------------------------------------------------
# CSV HEADERS
# ---------------------------------------------------------
PROCESS_CSV_HEADERS = [
    "timestamp",
    "pid",
    "process_name",
    "status",
    "cpu_percent",
    "memory_mb",
    "exe_path",
    "create_time",
    "username",
]

EVENT_CSV_HEADERS = [
    "timestamp",
    "event_type",     # CREATED or TERMINATED
    "pid",
    "process_name",
    "create_time",
]


def init_csv_file(file_path, headers):
    """Create the CSV file with headers if it does not already exist."""
    file_exists = os.path.isfile(file_path)
    if not file_exists:
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def get_process_snapshot():
    """
    Scan all currently running processes and return:
      1. A list of dictionaries with detailed process info
      2. A dictionary keyed by (pid, create_time) for change detection
    Using (pid, create_time) instead of just pid avoids false
    "same process" matches when a PID is reused by Windows.
    """
    processes_info = []
    current_keys = {}

    # Initialize per-process CPU measurement.
    # psutil requires a first call to cpu_percent() to "prime" the
    # measurement; the real value is only accurate on the *next* call.
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Small delay so cpu_percent() has a time window to measure against.
    time.sleep(0.2)

    for proc in psutil.process_iter():
        try:
            pid = proc.pid
            name = proc.name()
            status = proc.status()
            cpu = proc.cpu_percent(interval=None)

            try:
                mem_mb = proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                mem_mb = None

            try:
                exe_path = proc.exe()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                exe_path = "ACCESS_DENIED"

            try:
                create_time = proc.create_time()
                create_time_str = datetime.fromtimestamp(create_time).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                create_time = None
                create_time_str = "UNKNOWN"

            try:
                username = proc.username()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                username = "ACCESS_DENIED"

            info = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pid": pid,
                "process_name": name,
                "status": status,
                "cpu_percent": round(cpu, 2) if cpu is not None else None,
                "memory_mb": round(mem_mb, 2) if mem_mb is not None else None,
                "exe_path": exe_path,
                "create_time": create_time_str,
                "username": username,
            }

            processes_info.append(info)

            # Unique key handles PID reuse correctly
            key = (pid, create_time)
            current_keys[key] = info

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process disappeared mid-scan or we don't have permission.
            continue

    return processes_info, current_keys


def log_event(event_type, info, event_writer):
    """Write a CREATED or TERMINATED event to the event log."""
    event_writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        event_type,
        info.get("pid"),
        info.get("process_name"),
        info.get("create_time"),
    ])


def main():
    print("=" * 60)
    print("HyperMD-Enhanced - Process Collector (Module 1)")
    print("=" * 60)
    print(f"Scanning every {SCAN_INTERVAL_SECONDS} seconds.")
    print(f"Process data  -> {PROCESS_DATA_FILE}")
    print(f"Event log     -> {EVENT_LOG_FILE}")
    print("Press Ctrl + C to stop.\n")

    init_csv_file(PROCESS_DATA_FILE, PROCESS_CSV_HEADERS)
    init_csv_file(EVENT_LOG_FILE, EVENT_CSV_HEADERS)

    previous_keys = {}
    scan_count = 0

    try:
        with open(PROCESS_DATA_FILE, mode="a", newline="", encoding="utf-8") as pf, \
             open(EVENT_LOG_FILE, mode="a", newline="", encoding="utf-8") as ef:

            process_writer = csv.writer(pf)
            event_writer = csv.writer(ef)

            while True:
                processes_info, current_keys = get_process_snapshot()

                # Write full snapshot of current processes
                for info in processes_info:
                    process_writer.writerow([
                        info["timestamp"],
                        info["pid"],
                        info["process_name"],
                        info["status"],
                        info["cpu_percent"],
                        info["memory_mb"],
                        info["exe_path"],
                        info["create_time"],
                        info["username"],
                    ])
                pf.flush()

                # Detect newly created processes
                if previous_keys:  # skip event detection on the very first scan
                    new_keys = set(current_keys.keys()) - set(previous_keys.keys())
                    for key in new_keys:
                        log_event("CREATED", current_keys[key], event_writer)
                        print(f"[+] CREATED: PID {key[0]} - {current_keys[key]['process_name']}")

                    # Detect terminated processes
                    terminated_keys = set(previous_keys.keys()) - set(current_keys.keys())
                    for key in terminated_keys:
                        log_event("TERMINATED", previous_keys[key], event_writer)
                        print(f"[-] TERMINATED: PID {key[0]} - {previous_keys[key]['process_name']}")

                    ef.flush()

                previous_keys = current_keys
                scan_count += 1
                print(f"Scan #{scan_count} complete - {len(processes_info)} processes recorded.")

                time.sleep(SCAN_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl + C).")
        print(f"Total scans completed: {scan_count}")
        print(f"Data saved to: {os.path.abspath(PROCESS_DATA_FILE)}")
        print(f"Events saved to: {os.path.abspath(EVENT_LOG_FILE)}")


if __name__ == "__main__":
    main()