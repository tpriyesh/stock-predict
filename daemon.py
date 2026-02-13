#!/usr/bin/env python3
"""
Trading Daemon - Runs the trading agent automatically.

Features:
- Starts automatically at market open
- Stops after market close
- Low CPU usage when idle (sleeps)
- Handles laptop sleep/wake gracefully
- Auto-recovers from errors

Usage:
    python daemon.py start     # Start daemon
    python daemon.py stop      # Stop daemon
    python daemon.py status    # Check status
    python daemon.py run       # Run in foreground (for testing)
"""
import os
import sys
import time
import signal
import atexit
from datetime import datetime, time as dt_time, timedelta, date
from pathlib import Path
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

from utils.platform import now_ist, today_ist, time_ist, is_pid_running

# Load environment variables
load_dotenv()

# Paths
PID_FILE = Path.home() / ".trading_daemon.pid"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    LOG_DIR / "daemon_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)
# Issues-only log: WARNING and above — review at night to fix problems
logger.add(
    LOG_DIR / "issues_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="WARNING",
    format="{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)


class TradingDaemon:
    """
    Daemon process that runs the trading agent at market hours.
    All timing from CONFIG (env-driven).
    """

    def __init__(self):
        from config.trading_config import CONFIG
        self._config = CONFIG

        self._running = False
        self._orchestrator = None
        self._last_health_check = None

    def is_trading_day(self, check_date: date = None) -> bool:
        """Check if given date is a trading day."""
        check_date = check_date or today_ist()

        # Weekend check
        if check_date.weekday() >= 5:
            return False

        # Holiday check
        if check_date in self._config.holidays:
            return False

        return True

    def get_next_trading_day(self) -> date:
        """Get the next trading day."""
        check_date = today_ist()

        # If after market close, start checking from tomorrow
        if time_ist() > self._config.hours.post_market_end:
            check_date += timedelta(days=1)

        for _ in range(10):
            if self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)

        return check_date

    def seconds_until(self, target_time: dt_time, target_date: date = None) -> float:
        """Calculate seconds until target time."""
        target_date = target_date or today_ist()
        target_dt = datetime.combine(target_date, target_time)
        now = now_ist().replace(tzinfo=None)

        if target_dt < now:
            target_dt += timedelta(days=1)

        return (target_dt - now).total_seconds()

    def smart_sleep(self, seconds: float, max_chunk: float = 300) -> bool:
        """
        Sleep in chunks to allow for graceful shutdown.
        Uses wall-clock time so laptop sleep/wake is handled correctly.

        Args:
            seconds: Total seconds to sleep
            max_chunk: Maximum sleep chunk (5 minutes default)

        Returns:
            True if sleep completed, False if interrupted
        """
        deadline = time.monotonic() + seconds
        while self._running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            chunk = min(remaining, max_chunk)
            time.sleep(chunk)

            # Log status periodically
            remaining_after = deadline - time.monotonic()
            if remaining_after > 0 and int(remaining_after) % 3600 < max_chunk:
                logger.debug(f"Sleeping... {remaining_after/3600:.1f} hours remaining")

        return self._running

    def run_trading_session(self):
        """Run a single trading session."""
        logger.info("=" * 60)
        logger.info("STARTING TRADING SESSION")
        logger.info("=" * 60)

        try:
            # Initialize components
            mode = os.getenv("TRADING_MODE", "paper")
            logger.info(f"Trading mode: {mode}")

            from broker import get_broker
            from risk import RiskManager
            from agent.orchestrator import TradingOrchestrator

            broker = get_broker(mode)
            if not broker.connect():
                logger.error("Failed to connect to broker")
                return

            risk_manager = RiskManager(broker)

            # Check if we can trade
            can_trade, reason = risk_manager.can_trade()
            if not can_trade:
                logger.warning(f"Cannot trade: {reason}")
                return

            # Create and run orchestrator
            self._orchestrator = TradingOrchestrator(broker, risk_manager)

            # Run until market close or shutdown
            self._orchestrator.run(paper_mode=(mode == "paper"))

        except Exception as e:
            logger.exception(f"Trading session error: {e}")

        finally:
            self._orchestrator = None
            logger.info("Trading session ended")

    def run(self):
        """Main daemon loop."""
        self._running = True
        logger.info("Trading daemon started")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        while self._running:
            try:
                now = now_ist().replace(tzinfo=None)
                today = today_ist()

                # Check if today is a trading day
                if not self.is_trading_day(today):
                    next_day = self.get_next_trading_day()
                    sleep_until = datetime.combine(next_day, self._config.hours.pre_market_start)
                    sleep_seconds = (sleep_until - now).total_seconds()

                    logger.info(f"Not a trading day. Sleeping until {next_day} ({sleep_seconds/3600:.1f} hours)")
                    if not self.smart_sleep(sleep_seconds):
                        break
                    continue

                # Before market hours
                if now.time() < self._config.hours.pre_market_start:
                    sleep_seconds = self.seconds_until(self._config.hours.pre_market_start)
                    logger.info(f"Before market. Sleeping {sleep_seconds/60:.0f} minutes until pre-market")
                    if not self.smart_sleep(sleep_seconds):
                        break
                    continue

                # Pre-market preparation (8:45 - 9:15)
                if self._config.hours.pre_market_start <= now.time() < self._config.hours.market_open:
                    logger.info("Pre-market phase - preparing signals...")
                    self._prepare_signals()

                    sleep_seconds = self.seconds_until(self._config.hours.market_open)
                    logger.info(f"Waiting {sleep_seconds/60:.0f} minutes for market open")
                    if not self.smart_sleep(sleep_seconds):
                        break
                    continue

                # Market hours (9:15 - 15:30)
                if self._config.hours.market_open <= now.time() < self._config.hours.market_close:
                    logger.info("Market is OPEN - starting trading session")
                    self.run_trading_session()

                    # After session ends, wait for post-market
                    if time_ist() < self._config.hours.post_market_end:
                        sleep_seconds = self.seconds_until(self._config.hours.post_market_end)
                        logger.info(f"Post-market. Sleeping {sleep_seconds/60:.0f} minutes")
                        self.smart_sleep(sleep_seconds)
                    continue

                # After market hours
                if now.time() >= self._config.hours.market_close:
                    next_day = self.get_next_trading_day()
                    sleep_until = datetime.combine(next_day, self._config.hours.pre_market_start)
                    sleep_seconds = (sleep_until - now).total_seconds()

                    logger.info(f"Market closed. Next session: {next_day} ({sleep_seconds/3600:.1f} hours)")
                    if not self.smart_sleep(sleep_seconds):
                        break
                    continue

            except Exception as e:
                logger.exception(f"Daemon loop error: {e}")
                # Sleep before retry to prevent tight loop
                time.sleep(60)

        logger.info("Trading daemon stopped")

    def _prepare_signals(self):
        """Pre-market signal preparation."""
        try:
            from agent.signal_adapter import SignalAdapter
            adapter = SignalAdapter()
            signals = adapter.get_trade_signals(force_refresh=True)
            logger.info(f"Pre-market: Generated {len(signals)} signals")
        except Exception as e:
            logger.error(f"Failed to prepare signals: {e}")

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.warning(f"Received signal {signum}, shutting down...")
        self._running = False

        if self._orchestrator:
            self._orchestrator._shutdown_requested = True

    def stop(self):
        """Stop the daemon."""
        self._running = False


def daemonize():
    """Fork the process to run as daemon."""
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error(f"First fork failed: {e}")
        sys.exit(1)

    # Decouple from parent
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error(f"Second fork failed: {e}")
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    # Write PID file
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # Register cleanup
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))


def get_daemon_pid() -> Optional[int]:
    """Get the running daemon PID."""
    if not PID_FILE.exists():
        return None

    try:
        pid = int(PID_FILE.read_text().strip())
        if is_pid_running(pid):
            return pid
        PID_FILE.unlink(missing_ok=True)
        return None
    except (ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return None


def cmd_start():
    """Start the daemon."""
    pid = get_daemon_pid()
    if pid:
        print(f"Daemon already running (PID: {pid})")
        return

    print("Starting trading daemon...")

    # Fork to background
    daemonize()

    # Run daemon
    daemon = TradingDaemon()
    daemon.run()


def cmd_stop():
    """Stop the daemon."""
    pid = get_daemon_pid()
    if not pid:
        print("Daemon is not running")
        return

    print(f"Stopping daemon (PID: {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)

        # Wait for process to stop
        for _ in range(30):
            time.sleep(1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print("Daemon stopped")
                return

        # Force kill if still running
        os.kill(pid, signal.SIGKILL)
        print("Daemon force killed")

    except ProcessLookupError:
        print("Daemon already stopped")

    PID_FILE.unlink(missing_ok=True)


def cmd_status():
    """Check daemon status."""
    pid = get_daemon_pid()

    if pid:
        print(f"Daemon is RUNNING (PID: {pid})")

        # Show log tail
        log_file = LOG_DIR / f"daemon_{today_ist().strftime('%Y-%m-%d')}.log"
        if log_file.exists():
            print("\nRecent log entries:")
            print("-" * 40)
            lines = log_file.read_text().splitlines()
            for line in lines[-10:]:
                print(line)
    else:
        print("Daemon is NOT RUNNING")


def cmd_run():
    """Run in foreground (for testing)."""
    print("Running trading daemon in foreground...")
    print("Press Ctrl+C to stop\n")

    daemon = TradingDaemon()
    daemon.run()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trading Daemon")
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "run"],
        help="Daemon command"
    )

    args = parser.parse_args()

    if args.command == "start":
        cmd_start()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()
    elif args.command == "run":
        cmd_run()


if __name__ == "__main__":
    main()
