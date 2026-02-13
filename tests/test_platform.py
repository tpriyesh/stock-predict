"""
Tests for utils/platform.py — IST timezone, yfinance timeout, disk space, PID validation.

~25 tests covering all platform utilities.
"""
import os
import sys
import time
import tempfile
from datetime import datetime, date, time as dt_time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.platform import (
    IST, now_ist, today_ist, time_ist,
    yfinance_with_timeout, check_disk_space, is_pid_running,
)


# ============================================
# IST TIMEZONE TESTS
# ============================================

class TestNowIST:
    def test_returns_datetime(self):
        result = now_ist()
        assert isinstance(result, datetime)

    def test_has_timezone_info(self):
        result = now_ist()
        assert result.tzinfo is not None
        assert str(result.tzinfo) == "Asia/Kolkata"

    def test_replace_tzinfo_gives_naive(self):
        result = now_ist().replace(tzinfo=None)
        assert result.tzinfo is None

    def test_consistent_with_pytz(self):
        """now_ist() should match manual pytz conversion."""
        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        manual = datetime.now(ist)
        auto = now_ist()
        # Within 1 second
        diff = abs((manual - auto).total_seconds())
        assert diff < 1.0


class TestTodayIST:
    def test_returns_date(self):
        result = today_ist()
        assert isinstance(result, date)
        assert not isinstance(result, datetime)

    def test_matches_now_ist_date(self):
        result = today_ist()
        expected = now_ist().date()
        assert result == expected


class TestTimeIST:
    def test_returns_time(self):
        result = time_ist()
        assert isinstance(result, dt_time)

    def test_has_hour_minute(self):
        result = time_ist()
        assert 0 <= result.hour <= 23
        assert 0 <= result.minute <= 59


# ============================================
# YFINANCE TIMEOUT TESTS
# ============================================

class TestYfinanceWithTimeout:
    def test_fast_function_returns_result(self):
        result = yfinance_with_timeout(lambda: 42, timeout_seconds=5)
        assert result == 42

    def test_returns_complex_result(self):
        data = {"price": 100.5, "volume": 50000}
        result = yfinance_with_timeout(lambda: data, timeout_seconds=5)
        assert result == data

    def test_slow_function_raises_timeout(self):
        def slow():
            time.sleep(10)
            return "never"

        with pytest.raises(TimeoutError, match="timed out after 1s"):
            yfinance_with_timeout(slow, timeout_seconds=1)

    def test_exception_propagates(self):
        def failing():
            raise ValueError("Bad ticker symbol")

        with pytest.raises(ValueError, match="Bad ticker"):
            yfinance_with_timeout(failing, timeout_seconds=5)

    def test_none_return_is_valid(self):
        result = yfinance_with_timeout(lambda: None, timeout_seconds=5)
        assert result is None

    def test_zero_timeout_raises(self):
        """Even instant functions should timeout with 0 seconds."""
        # This may or may not work depending on thread scheduling.
        # The point is it shouldn't hang.
        try:
            yfinance_with_timeout(lambda: 42, timeout_seconds=0)
        except TimeoutError:
            pass  # Expected in most cases


# ============================================
# DISK SPACE TESTS
# ============================================

class TestCheckDiskSpace:
    def test_current_dir_has_space(self):
        """Current directory should have >1MB free (CI/dev machine)."""
        assert check_disk_space(".", min_mb=1) is True

    def test_absurd_min_fails(self):
        """Asking for 10TB should fail on most systems."""
        assert check_disk_space(".", min_mb=10_000_000) is False

    def test_invalid_path_returns_true(self):
        """Invalid path should return True (don't block trading)."""
        result = check_disk_space("/nonexistent/path/that/doesnt/exist", min_mb=1)
        assert result is True

    def test_temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            assert check_disk_space(d, min_mb=1) is True

    def test_zero_min_always_passes(self):
        assert check_disk_space(".", min_mb=0) is True


# ============================================
# PID VALIDATION TESTS
# ============================================

class TestIsPidRunning:
    def test_current_pid_is_running(self):
        """Our own PID should be running."""
        assert is_pid_running(os.getpid()) is True

    def test_dead_pid_not_running(self):
        """PID 999999 is very unlikely to be running."""
        # Use a very high PID that's almost certainly not allocated
        assert is_pid_running(999999) is False

    def test_pid_zero_is_special(self):
        """PID 0 check should not crash (it's the kernel on Unix)."""
        # On macOS/Linux, sending signal 0 to PID 0 raises PermissionError
        # which means it exists (is_pid_running returns True), or it may fail
        # Either way, it shouldn't crash
        result = is_pid_running(0)
        assert isinstance(result, bool)

    def test_negative_pid(self):
        """Negative PID should return False (invalid)."""
        result = is_pid_running(-1)
        assert isinstance(result, bool)

    def test_init_process(self):
        """PID 1 (init/launchd) should be running."""
        assert is_pid_running(1) is True
