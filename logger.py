"""
logger.py — tees stdout and stderr to both the terminal AND a log file.

Usage at the top of any script:
    from logger import setup_logging
    setup_logging("main")     # creates outputs/logs/main_YYYY-MM-DD_HH-MM-SS.log

Everything printed (including third-party library output, tqdm bars,
joblib progress, etc.) is also written to that file.
"""

import os
import sys
import time
from datetime import datetime

from paths import OUTPUT_BASE_PATH


class _Tee:
    """File-like object that writes to multiple streams at once."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    # tqdm checks for isatty() to decide whether to do nice progress bars
    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


_log_file_handle = None
_log_path = None


def setup_logging(name="run", folder=None):
    """Start logging stdout+stderr to a timestamped file.

    Returns the path to the log file.
    """
    global _log_file_handle, _log_path

    if folder is None:
        folder = os.path.join(OUTPUT_BASE_PATH, "logs")
    os.makedirs(folder, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _log_path = os.path.join(folder, f"{name}_{ts}.log")
    _log_file_handle = open(_log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    # Save original streams
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    # Replace with tees that write to both terminal and file
    sys.stdout = _Tee(orig_stdout, _log_file_handle)
    sys.stderr = _Tee(orig_stderr, _log_file_handle)

    # Print a startup banner
    banner = (
        f"{'='*70}\n"
        f"  LOG STARTED: {ts}\n"
        f"  Script:      {name}\n"
        f"  Log file:    {_log_path}\n"
        f"  Python:      {sys.version.split()[0]}\n"
        f"  CWD:         {os.getcwd()}\n"
        f"{'='*70}"
    )
    print(banner)

    return _log_path


def log_section(title):
    """Print a clear section header with a timestamp. Use to break up long logs."""
    ts = datetime.now().strftime("%H:%M:%S")
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  [{ts}]  {title}")
    print(f"{bar}")


def close_logging():
    """Close the log file. Optional — process exit also closes it."""
    global _log_file_handle
    if _log_file_handle:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log closed.")
        _log_file_handle.close()
        _log_file_handle = None
