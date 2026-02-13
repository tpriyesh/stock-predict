"""
Platform Utilities - IST timezone, yfinance timeout, disk check, PID validation.

Centralizes cross-cutting concerns so the rest of the codebase stays clean.
All timezone-sensitive code should use now_ist()/today_ist()/time_ist()
instead of datetime.now()/date.today() to ensure correct behavior on
VPS servers that may be in UTC or other timezones.
"""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, date, time
from pathlib import Path

import pytz
from loguru import logger


# ============================================
# IST TIMEZONE
# ============================================

IST = pytz.timezone('Asia/Kolkata')


def now_ist() -> datetime:
    """Get current datetime in IST regardless of server timezone."""
    return datetime.now(IST)


def today_ist() -> date:
    """Get today's date in IST."""
    return now_ist().date()


def time_ist() -> time:
    """Get current time-of-day in IST."""
    return now_ist().time()


# ============================================
# YFINANCE TIMEOUT
# ============================================

_YF_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yfinance")


def yfinance_with_timeout(func, timeout_seconds=15):
    """
    Execute a yfinance call with a timeout.

    yfinance doesn't support timeout natively, so we run it in a
    thread pool and raise TimeoutError if it takes too long.

    Args:
        func: Callable that performs the yfinance operation
        timeout_seconds: Max seconds to wait (default 15)

    Returns:
        Result of func()

    Raises:
        TimeoutError: If the call exceeds timeout_seconds
    """
    future = _YF_EXECUTOR.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeout:
        future.cancel()
        raise TimeoutError(f"yfinance call timed out after {timeout_seconds}s")


# ============================================
# DISK SPACE CHECK
# ============================================

def check_disk_space(path, min_mb=50) -> bool:
    """
    Check if there's enough disk space for SQLite writes.

    Args:
        path: Directory or file path to check
        min_mb: Minimum free space in MB (default 50MB)

    Returns:
        True if sufficient space available, True on error (don't block trading)
    """
    try:
        usage = shutil.disk_usage(str(path))
        free_mb = usage.free / (1024 * 1024)
        return free_mb >= min_mb
    except Exception:
        return True  # Don't block on check failure


# ============================================
# PID VALIDATION
# ============================================

def is_pid_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # Signal 0 = test if process exists
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but we don't own it
    except Exception:
        return False
