"""
logger.py — tees stdout/stderr to a timestamped log file, with:
  - banner that records hostname, OS, Python, CPU, RAM, free disk
  - uncaught-exception handler (full traceback written to log)
  - signal handlers for SIGTERM / SIGINT / SIGBREAK (Windows)
  - atexit + faulthandler dump on hard crashes
  - optional heartbeat thread that pings the log every N seconds
    so you can tell from the log timestamps when (if) the process died

Usage at the top of any script:
    from logger import setup_logging
    setup_logging("main", heartbeat_sec=300)
"""

import os
import sys
import time
import signal
import atexit
import platform
import threading
import traceback
import faulthandler
from datetime import datetime

from paths import OUTPUT_BASE_PATH


class _Tee:
    """File-like object that writes to multiple streams at once,
    prepending an ISO-style timestamp to each line.
    """
    def __init__(self, *streams, timestamp=True):
        self.streams = streams
        self.timestamp = timestamp
        self._at_line_start = True  # next write is the start of a new line

    def _prefix(self):
        return datetime.now().strftime("[%H:%M:%S] ")

    def write(self, data):
        if not data:
            return
        if not self.timestamp:
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass
            return

        # Build the output by injecting a timestamp at every line start.
        # We track _at_line_start across calls because tqdm and print can write
        # partial chunks (no trailing newline, then a newline later).
        out = []
        for ch in data:
            if self._at_line_start and ch not in ("\n", "\r"):
                out.append(self._prefix())
                self._at_line_start = False
            out.append(ch)
            if ch == "\n":
                self._at_line_start = True
        stamped = "".join(out)

        for s in self.streams:
            try:
                s.write(stamped)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


# Module-level state
_log_file_handle = None
_log_path = None
_heartbeat_thread = None
_heartbeat_stop = threading.Event()


def _system_info():
    """Collect a snapshot of the environment for the log banner."""
    info = {}
    info["hostname"]   = platform.node()
    info["platform"]   = platform.platform()
    info["python"]     = sys.version.split()[0]
    info["executable"] = sys.executable
    info["cwd"]        = os.getcwd()
    info["pid"]        = os.getpid()
    info["argv"]       = " ".join(sys.argv)

    # Optional psutil — only if installed
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["cpu_count"]    = f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical"
        info["ram_total_gb"] = f"{vm.total / 1e9:.1f}"
        info["ram_avail_gb"] = f"{vm.available / 1e9:.1f}"
        info["disk_free_gb"] = f"{psutil.disk_usage(os.getcwd()).free / 1e9:.1f}"
    except ImportError:
        info["cpu_count"] = os.cpu_count()
    return info


def _excepthook(exc_type, exc_value, exc_tb):
    """Catch uncaught exceptions and write full traceback to the log."""
    print("\n" + "!" * 70, flush=True)
    print(f"!!  UNCAUGHT EXCEPTION at {datetime.now().isoformat()}", flush=True)
    print("!" * 70, flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("!" * 70, flush=True)
    # Then chain to default handler so the process exits with proper code
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _signal_handler(signum, frame):
    """Catch termination signals (Ctrl+C, taskkill, OS shutdown notice)."""
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    msg = (
        f"\n{'#' * 70}\n"
        f"##  RECEIVED SIGNAL {signum} ({sig_name}) at {datetime.now().isoformat()}\n"
        f"##  This usually means: Ctrl+C, taskkill, or system shutdown.\n"
        f"##  Closing log and exiting.\n"
        f"{'#' * 70}"
    )
    print(msg, flush=True)
    close_logging()
    # Exit with conventional code 128 + signum
    sys.exit(128 + signum)


def _heartbeat_loop(interval):
    """Write a one-line heartbeat every `interval` seconds.
    If the log just stops, you can see exactly when it died.
    """
    while not _heartbeat_stop.wait(interval):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[heartbeat {ts}]", flush=True)
        except Exception:
            pass


def setup_logging(name="run", folder=None, heartbeat_sec=300):
    """Start logging. heartbeat_sec=0 disables the heartbeat thread."""
    global _log_file_handle, _log_path, _heartbeat_thread

    if folder is None:
        folder = os.path.join(OUTPUT_BASE_PATH, "logs")
    os.makedirs(folder, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _log_path = os.path.join(folder, f"{name}_{ts}.log")
    _log_file_handle = open(_log_path, "w", encoding="utf-8", buffering=1)  # line-buffered

    # Tee stdout and stderr to file
    sys.stdout = _Tee(sys.__stdout__, _log_file_handle)
    sys.stderr = _Tee(sys.__stderr__, _log_file_handle)

    # Dump C-level crash tracebacks to the log too (catches segfaults, MemoryErrors, etc.)
    try:
        faulthandler.enable(file=_log_file_handle)
    except Exception:
        pass

    # Catch uncaught Python exceptions
    sys.excepthook = _excepthook

    # Catch OS signals
    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _signal_handler)
            except Exception:
                pass  # SIGBREAK only on Windows, SIGHUP only on Unix, etc.

    # Make sure log gets flushed/closed when process exits cleanly
    atexit.register(close_logging)

    # Banner
    info = _system_info()
    banner_lines = [
        "=" * 70,
        f"  LOG STARTED: {ts}",
        f"  Script:      {name}",
        f"  Log file:    {_log_path}",
        f"  PID:         {info['pid']}",
        f"  Hostname:    {info['hostname']}",
        f"  Platform:    {info['platform']}",
        f"  Python:      {info['python']}",
        f"  Executable:  {info['executable']}",
        f"  CWD:         {info['cwd']}",
        f"  argv:        {info['argv']}",
        f"  CPU:         {info['cpu_count']}",
    ]
    if "ram_total_gb" in info:
        banner_lines.append(f"  RAM:         {info['ram_total_gb']} GB total, {info['ram_avail_gb']} GB available")
        banner_lines.append(f"  Disk free:   {info['disk_free_gb']} GB")
    banner_lines.append("=" * 70)
    print("\n".join(banner_lines))

    # Start heartbeat
    if heartbeat_sec and heartbeat_sec > 0:
        _heartbeat_stop.clear()
        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop, args=(heartbeat_sec,),
            daemon=True, name="logger-heartbeat",
        )
        _heartbeat_thread.start()
        print(f"[heartbeat enabled, every {heartbeat_sec}s]")

    return _log_path


def log_section(title):
    """Print a clear section header with a timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  [{ts}]  {title}")
    print(bar)


def close_logging():
    """Stop heartbeat, flush, and close log file. Idempotent."""
    global _log_file_handle, _heartbeat_thread
    if _heartbeat_thread is not None:
        _heartbeat_stop.set()
        _heartbeat_thread = None
    if _log_file_handle:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log closed cleanly.")
        except Exception:
            pass
        try:
            _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None