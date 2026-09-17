"""
HyperMD-Enhanced - Benign Stress Test Script
=============================================
Safely and deliberately spikes CPU and memory usage for a controlled
period, purely to generate a "high resource usage" test scenario for
labeling as class 1 in the CNN/LSTM pipeline.

This script does nothing malicious: no file encryption, no network
activity, no persistence, no system modification. It only consumes
CPU cycles and allocates memory, then releases everything and exits.

Usage examples
--------------
# Default: 60-second timed run (no interaction needed)
    python test_scripts/benign_stress_test.py

# Custom duration and memory
    python test_scripts/benign_stress_test.py --duration 120 --memory 300 --workers 4

# Interactive mode (waits for ENTER so you can dump the process)
    python test_scripts/benign_stress_test.py --interactive

# Print help
    python test_scripts/benign_stress_test.py --help

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\test_scripts\\benign_stress_test.py
"""

import argparse
import multiprocessing
import os
import sys
import threading
import time


# ---------------------------------------------------------
# DEFAULT CONFIGURATION (override via CLI args)
# ---------------------------------------------------------
DEFAULT_DURATION_SECONDS = 60   # total run time in timed mode
DEFAULT_WORKERS = multiprocessing.cpu_count()  # one worker per logical core
DEFAULT_MEMORY_MB = 200         # MB to allocate and hold


# ---------------------------------------------------------
# WORKER: CPU burn (runs in a child process)
# ---------------------------------------------------------
def _cpu_burn_worker(duration: float, worker_id: int) -> None:
    """Keep one CPU core busy with arithmetic for `duration` seconds."""
    end_time = time.monotonic() + duration
    x = 0.0
    while time.monotonic() < end_time:
        # Trivial busy-work - no meaningful computation
        x = (x + 1.23456) * 0.99999
    # Worker exits cleanly; no output here to avoid interleaved prints


# ---------------------------------------------------------
# MEMORY: Allocate and hold in the main process
# ---------------------------------------------------------
def _allocate_and_hold(size_mb: int, stop_event: threading.Event) -> None:
    """
    Allocate `size_mb` MB, touch every page so the OS commits it,
    then hold until stop_event is set.
    """
    num_bytes = size_mb * 1024 * 1024
    print(f"  Allocating {size_mb} MB ... ", end="", flush=True)
    try:
        block = bytearray(num_bytes)
    except MemoryError:
        print(f"\n[ERROR] Not enough RAM to allocate {size_mb} MB. Reduce --memory.")
        stop_event.set()
        return

    # Touch every page (4 KB) to force the OS to actually commit the pages
    for i in range(0, num_bytes, 4096):
        block[i] = 0xAB

    print("done.")
    print(f"  Memory is held. Waiting for test to complete...\n")

    # Hold until the orchestrator signals us
    stop_event.wait()

    print("  Releasing memory...")
    del block


# ---------------------------------------------------------
# LIVE STATUS TICKER (runs in a background thread)
# ---------------------------------------------------------
def _status_ticker(stop_event: threading.Event, duration: float | None) -> None:
    """Print a live one-line status every second."""
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False

    start = time.monotonic()
    while not stop_event.is_set():
        elapsed = time.monotonic() - start
        if has_psutil:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            mem_used = mem.used // (1024 * 1024)
            mem_pct = mem.percent
            status = (
                f"  [{elapsed:6.1f}s] CPU: {cpu:5.1f}%  "
                f"RAM used: {mem_used} MB ({mem_pct:.1f}%)"
            )
        else:
            status = f"  [{elapsed:6.1f}s] (install psutil for live stats)"

        if duration:
            remaining = max(0.0, duration - elapsed)
            status += f"  | remaining: {remaining:.0f}s"

        # Overwrite the same line
        print(f"\r{status}   ", end="", flush=True)
        time.sleep(1.0)

    print()  # newline after the ticker line ends


# ---------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------
def run_stress_test(
    duration: float | None,
    num_workers: int,
    memory_mb: int,
    interactive: bool,
) -> None:
    pid = os.getpid()

    print("=" * 62)
    print("  HyperMD-Enhanced -- Benign Stress Test")
    print("=" * 62)
    print(f"  Main PID   : {pid}")
    print(f"  CPU workers: {num_workers}")
    print(f"  Memory     : {memory_mb} MB")
    if interactive:
        print("  Mode       : INTERACTIVE (press ENTER when done dumping)")
    else:
        print(f"  Mode       : TIMED ({duration:.0f} seconds)")
    print()

    # --- Start CPU workers ---
    worker_burn_time = (duration or 9999) + 10  # give workers a bit of headroom
    workers: list[multiprocessing.Process] = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=_cpu_burn_worker,
            args=(worker_burn_time, i),
            daemon=True,   # daemon=True so they die if the main process crashes
            name=f"cpu-worker-{i}",
        )
        p.start()
        workers.append(p)
        print(f"  CPU worker {i} started  (PID {p.pid})")

    print()

    # --- Allocate memory in a background thread ---
    mem_stop = threading.Event()
    mem_thread = threading.Thread(
        target=_allocate_and_hold,
        args=(memory_mb, mem_stop),
        daemon=True,
        name="memory-holder",
    )
    mem_thread.start()
    mem_thread.join(timeout=15)  # wait up to 15 s for allocation to finish

    if mem_stop.is_set():
        # Allocation failed - clean up and exit
        for p in workers:
            p.terminate()
        sys.exit(1)

    # --- Live status ticker ---
    ticker_stop = threading.Event()
    ticker = threading.Thread(
        target=_status_ticker,
        args=(ticker_stop, duration),
        daemon=True,
        name="status-ticker",
    )
    ticker.start()

    # --- Wait phase ---
    try:
        if interactive:
            # Pause the ticker temporarily so input() doesn't get mangled
            ticker_stop.set()
            ticker.join()
            print(f"\n  Use PID {pid} with:  python collectors/dump_process.py {pid}")
            input("\n>>> Press ENTER when you have finished dumping this process ... ")
        else:
            time.sleep(duration)
    except KeyboardInterrupt:
        print("\n\n  [Ctrl+C] Interrupted by user.")

    # --- Tear down ---
    ticker_stop.set()
    ticker.join(timeout=2)

    mem_stop.set()          # signal memory thread to release
    mem_thread.join(timeout=5)

    print("\n  Terminating CPU workers...")
    for p in workers:
        p.terminate()
    for p in workers:
        p.join(timeout=5)

    print("  All workers stopped.")
    print("\n  Stress test complete. All resources released.")
    print("=" * 62)


# ---------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HyperMD-Enhanced Benign Stress Test -- spikes CPU & RAM safely.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        metavar="SECONDS",
        help="How long to run the stress test (timed mode).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help="Number of CPU-burning worker processes.",
    )
    parser.add_argument(
        "--memory", "-m",
        type=int,
        default=DEFAULT_MEMORY_MB,
        metavar="MB",
        help="MB of RAM to allocate and hold during the test.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=False,
        help=(
            "Interactive mode: pause and wait for ENTER instead of "
            "auto-exiting after --duration. Useful for ProcDump workflows."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    # REQUIRED on Windows: prevents child processes from re-running main()
    multiprocessing.freeze_support()

    args = _parse_args()
    run_stress_test(
        duration=args.duration,
        num_workers=args.workers,
        memory_mb=args.memory,
        interactive=args.interactive,
    )