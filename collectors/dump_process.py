"""
HyperMD-Enhanced - Module 2C: Memory Dump Trigger
Wraps Sysinternals ProcDump to capture a full memory dump of a given PID.
Run this on the Windows 10 VM, with ProcDump already installed.

Save at: D:\Rokesh Project\HyperMD-Enhanced\collectors\dump_process.py
"""

import subprocess
import os
import sys
from datetime import datetime

PROCDUMP_PATH = r"C:\Tools\ProcDump\procdump.exe"
DUMP_FOLDER = r"D:\Rokesh Project\HyperMD-Enhanced\memory_dumps"


def dump_process(pid: int):
    if not os.path.isfile(PROCDUMP_PATH):
        print(f"ERROR: ProcDump not found at {PROCDUMP_PATH}")
        return

    os.makedirs(DUMP_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(DUMP_FOLDER, f"process_{pid}_{timestamp}.dmp")

    command = [PROCDUMP_PATH, "-ma", "-accepteula", str(pid), output_file]

    print(f"Dumping PID {pid} -> {output_file}")
    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # ProcDump's exit code does not follow the simple 0 = success
    # convention, so check for the actual dump file instead.
    if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
        size_mb = round(os.path.getsize(output_file) / (1024 * 1024), 2)
        print(f"Dump completed successfully. File size: {size_mb} MB")
    else:
        print("ERROR: Dump file was not created. Check ProcDump output above.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dump_process.py <PID>")
        sys.exit(1)

    dump_process(int(sys.argv[1]))