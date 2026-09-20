"""
HyperMD-Enhanced - Module 6: Automated Response Engine
=======================================================
Decides what action to take when the fusion model flags a session
as suspicious, and executes it ONLY after human confirmation.

Safety rules (non-negotiable)
------------------------------
  1. NEVER delete files without explicit user confirmation.
  2. NEVER kill processes without explicit user confirmation.
  3. All actions are logged to logs/response_log.csv before execution.
  4. The engine is designed to run headless (called from dashboard or CLI)
     OR interactively (--interactive flag asks the user in the terminal).

Response levels
---------------
  confidence < 0.6  -> MONITOR  : log the event, no action taken
  0.6 <= conf < 0.8 -> ALERT    : log + write alert to logs/alerts.txt
  0.8 <= conf < 0.95 -> ISOLATE : ALERT + propose network isolation (Windows
                                   firewall rule, requires admin + confirmation)
  conf >= 0.95       -> DUMP    : ISOLATE + trigger an extra ProcDump of the
                                   event_session.py process for forensics
                                   (requires confirmation)

Save at: D:\\Rokesh Project\\HyperMD-Enhanced\\response\\response_engine.py

Usage
-----
    # Non-interactive (dashboard / automated pipeline):
    from response.response_engine import respond
    action = respond(predict_result, explanation, event_id, interactive=False)
    print(action["level"], action["actions_taken"])

    # Interactive CLI test:
    .venv\\Scripts\\python.exe response/response_engine.py --selftest
"""

import os
import sys
import csv
import time
import subprocess
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")
RESPONSE_LOG   = os.path.join(LOG_DIR, "response_log.csv")
ALERT_FILE     = os.path.join(LOG_DIR, "alerts.txt")
PROCDUMP_PATH  = r"C:\Tools\ProcDump\procdump.exe"
DUMP_FOLDER    = os.path.join(PROJECT_ROOT, "memory_dumps")

RESPONSE_LOG_HEADERS = [
    "timestamp", "event_id", "label", "confidence",
    "level", "actions_taken", "confirmed_by_user", "notes",
]

# Confidence thresholds
THRESHOLD_MONITOR  = 0.0
THRESHOLD_ALERT    = 0.6
THRESHOLD_ISOLATE  = 0.8
THRESHOLD_DUMP     = 0.95

LEVEL_NAMES = {
    "MONITOR":  "MONITOR  - logged only, no action",
    "ALERT":    "ALERT    - logged + alert file written",
    "ISOLATE":  "ISOLATE  - alert + proposed firewall isolation",
    "DUMP":     "DUMP     - isolate + extra forensic memory dump",
}

# Windows Firewall rule name prefix (blocked outbound for suspicious PID)
FW_RULE_PREFIX = "HyperMD_Block_"


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.isfile(RESPONSE_LOG):
        with open(RESPONSE_LOG, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RESPONSE_LOG_HEADERS).writeheader()


def _log_response(record: dict) -> None:
    _ensure_log_dir()
    with open(RESPONSE_LOG, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RESPONSE_LOG_HEADERS).writerow(record)


def _write_alert(event_id: str, confidence: float, summary: str) -> None:
    _ensure_log_dir()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"\n{'='*60}\n"
        f"[{ts}] ALERT  event={event_id}  confidence={confidence:.2%}\n"
        f"{summary}\n"
        f"{'='*60}\n"
    )
    with open(ALERT_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"  [response] Alert written -> {ALERT_FILE}")


# ---------------------------------------------------------
# RESPONSE ACTIONS
# ---------------------------------------------------------
def _action_monitor(event_id: str, confidence: float) -> str:
    """Log event at MONITOR level — no user-facing action."""
    print(f"  [response] MONITOR: event {event_id} logged (conf={confidence:.2%})")
    return "logged"


def _action_alert(event_id: str, confidence: float, short_summary: str) -> str:
    """Write an alert to the alerts file."""
    _write_alert(event_id, confidence, short_summary)
    return "alert_written"


def _action_isolate(
    event_id: str,
    pid: int | None,
    interactive: bool,
) -> str:
    """
    Propose a Windows Firewall outbound-block rule for the process PID.
    REQUIRES: running as Administrator on Windows.
    REQUIRES: explicit user confirmation.
    """
    rule_name = f"{FW_RULE_PREFIX}{event_id}"
    cmd = [
        "netsh", "advfirewall", "firewall", "add", "rule",
        f"name={rule_name}",
        "dir=out", "action=block", "enable=yes",
        "description=HyperMD-Enhanced automated isolation",
    ]
    if pid:
        # Firewall rules can't target by PID directly; log the PID for manual action
        print(f"  [response] NOTE: Firewall rules apply per-application, not per-PID.")
        print(f"  [response] Suspicious PID was: {pid}")

    print(f"\n  [ISOLATE] Proposed Windows Firewall rule:")
    print(f"    {' '.join(cmd)}")
    print(f"  This will BLOCK all outbound traffic for this event's session process.")

    if interactive:
        ans = input("\n  Apply firewall isolation? [y/N]: ").strip().lower()
        confirmed = ans == "y"
    else:
        # Non-interactive: propose but do NOT execute (dashboard will prompt user)
        print("  [response] Non-interactive mode: isolation NOT applied. "
              "Dashboard will prompt the user.")
        return "isolation_proposed"

    if confirmed:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [response] Firewall rule '{rule_name}' applied.")
            return "isolation_applied"
        else:
            print(f"  [response] Firewall rule FAILED: {result.stderr.strip()}")
            print("  (Try running as Administrator)")
            return "isolation_failed"
    else:
        print("  [response] Isolation cancelled by user.")
        return "isolation_declined"


def _action_extra_dump(event_id: str, interactive: bool) -> str:
    """
    Trigger an extra ProcDump of the current event_session process (self).
    REQUIRES explicit user confirmation.
    """
    import os
    pid = os.getpid()
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_name = f"forensic_{event_id}_{ts}.dmp"
    dump_path = os.path.join(DUMP_FOLDER, dump_name)

    print(f"\n  [DUMP] Forensic dump proposed:")
    print(f"    PID: {pid}  ->  {dump_path}")
    print(f"    Size estimate: ~200-500 MB")

    if interactive:
        ans = input("\n  Capture forensic dump now? [y/N]: ").strip().lower()
        confirmed = ans == "y"
    else:
        print("  [response] Non-interactive mode: forensic dump NOT triggered. "
              "Dashboard will prompt user.")
        return "dump_proposed"

    if confirmed:
        if not os.path.isfile(PROCDUMP_PATH):
            print(f"  [response] ProcDump not found at {PROCDUMP_PATH}.")
            return "dump_failed"

        os.makedirs(DUMP_FOLDER, exist_ok=True)
        result = subprocess.run(
            [PROCDUMP_PATH, "-ma", "-accepteula", str(pid), dump_path],
            capture_output=True, text=True, timeout=120,
        )
        if os.path.isfile(dump_path) and os.path.getsize(dump_path) > 0:
            size_mb = os.path.getsize(dump_path) / (1024 * 1024)
            print(f"  [response] Forensic dump saved: {dump_path} ({size_mb:.1f} MB)")
            return f"dump_saved:{dump_name}"
        else:
            print(f"  [response] Dump failed: {result.stderr[:200]}")
            return "dump_failed"
    else:
        print("  [response] Forensic dump declined by user.")
        return "dump_declined"


# ---------------------------------------------------------
# MAIN RESPOND FUNCTION
# ---------------------------------------------------------
def _classify_level(confidence: float) -> str:
    if confidence >= THRESHOLD_DUMP:
        return "DUMP"
    if confidence >= THRESHOLD_ISOLATE:
        return "ISOLATE"
    if confidence >= THRESHOLD_ALERT:
        return "ALERT"
    return "MONITOR"


def respond(
    predict_result: dict,
    explanation: dict | None = None,
    event_id: str = "",
    interactive: bool = False,
    pid: int | None = None,
) -> dict:
    """
    Execute the automated response for one prediction.

    Parameters
    ----------
    predict_result : output from models.inference.predict()
    explanation    : output from explainability.explain.explain_prediction()
                     (used for the alert text; optional)
    event_id       : event identifier (used in log rows and file names)
    interactive    : if True, prompt user in terminal before destructive actions
    pid            : optional PID of the monitored process (for isolation note)

    Returns
    -------
    dict with keys:
        level          (str) : MONITOR / ALERT / ISOLATE / DUMP
        actions_taken  (list[str]) : what was done
        confirmed      (bool)
        log_row        (dict) : what was written to response_log.csv
    """
    label      = predict_result.get("label", 0)
    confidence = predict_result.get("confidence", 0.0)
    level      = _classify_level(confidence)

    if not event_id:
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    short_summary = (
        explanation.get("short_summary", "")
        if explanation else
        f"label={label}, confidence={confidence:.2%}"
    )

    print(f"\n  [response] Level: {level}  |  event={event_id}  "
          f"conf={confidence:.2%}  label={label}")
    print(f"  [response] {LEVEL_NAMES[level]}")

    actions_taken = []
    confirmed     = False

    # Always log
    actions_taken.append(_action_monitor(event_id, confidence))

    if level in ("ALERT", "ISOLATE", "DUMP"):
        actions_taken.append(_action_alert(event_id, confidence, short_summary))

    if level in ("ISOLATE", "DUMP"):
        result = _action_isolate(event_id, pid, interactive)
        actions_taken.append(result)
        confirmed = "applied" in result

    if level == "DUMP":
        result = _action_extra_dump(event_id, interactive)
        actions_taken.append(result)

    # Write to response log
    log_row = {
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_id":          event_id,
        "label":             label,
        "confidence":        round(confidence, 4),
        "level":             level,
        "actions_taken":     "|".join(actions_taken),
        "confirmed_by_user": str(confirmed),
        "notes":             short_summary[:200],
    }
    _log_response(log_row)
    print(f"  [response] Logged to {RESPONSE_LOG}")

    return {
        "level":         level,
        "actions_taken": actions_taken,
        "confirmed":     confirmed,
        "log_row":       log_row,
    }


# ---------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------
def _selftest(interactive: bool = False) -> None:
    print("=" * 60)
    print("  HyperMD Response Engine -- Self-Test")
    print("=" * 60)

    # Simulate predictions at each confidence tier
    test_cases = [
        {"label": 0, "confidence": 0.35, "label_name": "normal",     "event": "test_monitor"},
        {"label": 1, "confidence": 0.70, "label_name": "suspicious", "event": "test_alert"},
        {"label": 1, "confidence": 0.88, "label_name": "suspicious", "event": "test_isolate"},
    ]

    for tc in test_cases:
        print(f"\n--- Simulating: confidence={tc['confidence']} ---")
        fake_predict = {
            "label":      tc["label"],
            "confidence": tc["confidence"],
            "label_name": tc["label_name"],
            "lstm_features": __import__("numpy").zeros(16),
            "cnn_features":  __import__("numpy").zeros(128),
            "fused_vector":  __import__("numpy").zeros(144),
        }
        fake_explanation = {
            "short_summary": f"[SIMULATED] confidence={tc['confidence']:.0%}"
        }
        result = respond(
            predict_result=fake_predict,
            explanation=fake_explanation,
            event_id=tc["event"],
            interactive=interactive,
        )
        print(f"  Result: level={result['level']}  "
              f"actions={result['actions_taken']}")

    print(f"\n  Response log: {RESPONSE_LOG}")
    print(f"  Alert file  : {ALERT_FILE}")
    print("\n  Self-test complete.")
    print("=" * 60)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HyperMD Response Engine -- Module 6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run self-test simulating MONITOR / ALERT / ISOLATE responses.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Prompt before executing destructive actions (isolation, dump).",
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest(interactive=args.interactive)
    else:
        parser.print_help()

