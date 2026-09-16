
import csv
import os
import time
from datetime import datetime

import psutil

OUTPUT_FILE = "performance_data.csv"
INTERVAL = 1  # Collect every 1 second


def collect_data():
    memory = psutil.virtual_memory()
    disk = psutil.disk_io_counters()

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "memory_percent": memory.percent,
        "process_count": len(psutil.pids()),
        "disk_read_bytes": disk.read_bytes if disk else 0,
        "disk_write_bytes": disk.write_bytes if disk else 0,
    }


def main():
    file_exists = os.path.exists(OUTPUT_FILE)

    fieldnames = [
        "timestamp",
        "cpu_percent",
        "memory_percent",
        "process_count",
        "disk_read_bytes",
        "disk_write_bytes",
    ]

    with open(OUTPUT_FILE, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        print("HyperMD Performance Collector Started")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                row = collect_data()
                writer.writerow(row)
                file.flush()

                print(
                    f"CPU: {row['cpu_percent']:5.1f}% | "
                    f"Memory: {row['memory_percent']:5.1f}% | "
                    f"Processes: {row['process_count']}"
                )

                time.sleep(INTERVAL)

        except KeyboardInterrupt:
            print("\nCollector stopped.")


if __name__ == "__main__":
    main()