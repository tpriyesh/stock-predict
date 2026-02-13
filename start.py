#!/usr/bin/env python3
"""
Daily Trading Start Script - Run this every morning.

This is the ONLY command you need to run each day:

    python start.py              # Paper trading (safe)
    python start.py --live       # Real money (uses BROKER env var)
    python start.py --zerodha    # Real money with Zerodha

What it does:
    1. Acquires file lock (prevents double instances)
    2. Checks broker token (auto-refreshes Zerodha via TOTP)
    3. Waits for market to open (if started early)
    4. Runs trading session (9:15 AM - 3:30 PM)
    5. Exits all positions before market close
    6. Prints daily report
    7. Exits cleanly

You can safely close with Ctrl+C at any time.
All positions will be exited before shutdown.
"""
import os
import sys
import time
import signal
import fcntl
from datetime import datetime, time as dt_time, date, timedelta
from pathlib import Path

from utils.platform import now_ist, today_ist, time_ist, is_pid_running

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

logger.add(
    "logs/trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)
# Issues-only log: WARNING and above — review at night to fix problems
logger.add(
    "logs/issues_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="WARNING",
    format="{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)


from config.trading_config import CONFIG


# ============================================
# SECURITY CHECKS
# ============================================

def check_env_permissions():
    """Warn if .env file has overly permissive permissions."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    try:
        import stat
        mode = env_path.stat().st_mode
        # Warn if group or others can read .env (contains API keys)
        if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            oct_mode = oct(mode & 0o777)
            logger.warning(
                f".env file has permissive permissions ({oct_mode}). "
                f"Run: chmod 600 {env_path}"
            )
            print(f"  WARNING: .env is world/group-readable ({oct_mode})")
            print(f"  Fix: chmod 600 {env_path}")
            print()
    except Exception:
        pass  # Skip on platforms without Unix permissions


# ============================================
# FILE LOCK - Prevents double instances
# ============================================

LOCK_FILE = Path(__file__).parent / "data" / ".trading.lock"


class FileLock:
    """File-based lock to prevent multiple trading instances."""

    def __init__(self, lock_path: Path = LOCK_FILE):
        self.lock_path = lock_path
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = None

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns False if another instance is running."""
        try:
            self._fd = open(self.lock_path, 'w')
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd.write(f"PID: {os.getpid()}\nStarted: {now_ist().isoformat()}\n")
            self._fd.flush()
            return True
        except (IOError, OSError):
            if self._fd:
                self._fd.close()
                self._fd = None
            return False

    def release(self):
        """Release the lock."""
        if self._fd:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
            except Exception:
                pass
            self._fd = None
            try:
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Another trading instance is already running!")
        return self

    def __exit__(self, *args):
        self.release()


# ============================================
# STARTUP CHECKS
# ============================================

def print_banner():
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  STOCK-PREDICT TRADING AGENT")
    print("=" * 60)
    print(f"  Date:    {today_ist().strftime('%A, %B %d, %Y')}")
    print(f"  Time:    {now_ist().strftime('%H:%M:%S')} IST")
    print("=" * 60)
    print()


def check_trading_day() -> bool:
    """Check if today is a trading day."""
    today = today_ist()

    if today.weekday() >= 5:
        print(f"Today is {today.strftime('%A')} - market closed on weekends.")
        return False

    if today in CONFIG.holidays:
        print("Today is a market holiday - no trading.")
        return False

    return True


def check_upstox_token(live: bool) -> bool:
    """Check if Upstox token is valid."""
    if not live:
        return True  # Paper mode doesn't need Upstox

    token = os.getenv("UPSTOX_ACCESS_TOKEN", "")

    if not token:
        print("No Upstox access token found!")
        print()
        print("Run this first:")
        print("  python scripts/upstox_auth.py")
        print()
        return False

    # Test the token
    print("Checking Upstox connection...", end=" ", flush=True)

    try:
        import requests
        response = requests.get(
            "https://api.upstox.com/v2/user/profile",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                user = data.get("data", {})
                print(f"OK ({user.get('user_name', 'Connected')})")
                return True

        print("FAILED")
        print()
        print("Token expired or invalid. Run:")
        print("  python scripts/upstox_auth.py")
        print()
        return False

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def check_zerodha_token(live: bool) -> bool:
    """Check if Zerodha token is valid, auto-refresh via TOTP if expired."""
    if not live:
        return True  # Paper mode doesn't need Zerodha

    print("Checking Zerodha connection...", end=" ", flush=True)

    # First try the current token
    token = os.getenv("ZERODHA_ACCESS_TOKEN", "")
    api_key = os.getenv("ZERODHA_API_KEY", "")

    if token and api_key:
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(token)
            profile = kite.profile()
            print(f"OK ({profile.get('user_name', 'Connected')})")
            return True
        except Exception:
            print("Token expired, refreshing...")

    # Auto-refresh via TOTP login
    print("  Running automated TOTP login...")
    try:
        from scripts.zerodha_auto_login import perform_login, _save_token
        access_token = perform_login()
        _save_token(access_token)
        # Update env for this process
        os.environ["ZERODHA_ACCESS_TOKEN"] = access_token
        print("  Token refreshed successfully!")
        return True
    except Exception as e:
        print(f"  Auto-login FAILED: {e}")
        print()
        print("  Run setup first:")
        print("    python scripts/zerodha_setup.py")
        print()
        return False


def _resolve_broker_type(args) -> str:
    """Determine which live broker to use based on flags and env var.

    Priority: --zerodha flag > BROKER env var > default 'upstox'
    """
    if getattr(args, 'zerodha', False):
        return 'zerodha'
    return os.getenv('BROKER', 'upstox').lower()


def check_capital(broker) -> bool:
    """Check if capital is above hard stop before starting."""
    try:
        from risk import RiskManager
        rm = RiskManager(broker)
        pv = rm.get_portfolio_value()
        if pv < CONFIG.capital.hard_stop_loss:
            print(f"\nPortfolio value Rs.{pv:,.0f} is BELOW hard stop Rs.{CONFIG.capital.hard_stop_loss:,.0f}")
            print("Cannot start trading. Add capital or reset kill switch.")
            return False
        return True
    except Exception:
        return True  # Don't block on check failure


def wait_for_market():
    """Wait for market to open if started early."""
    now = time_ist()

    if now >= CONFIG.hours.market_close:
        print("Market is already closed for today.")
        return False

    if now < CONFIG.hours.pre_market_start:
        target = datetime.combine(today_ist(), CONFIG.hours.pre_market_start)
        wait_seconds = (target - now_ist().replace(tzinfo=None)).total_seconds()
        minutes = wait_seconds / 60

        open_time = CONFIG.hours.market_open.strftime('%H:%M')
        print(f"Market opens at {open_time} IST. Waiting {minutes:.0f} minutes...")
        print("(sleeping - very low CPU usage)")
        print()

        while time_ist() < CONFIG.hours.pre_market_start:
            time.sleep(30)

    return True


# ============================================
# TRADING SESSION
# ============================================

def run_trading_session(live: bool, broker_type: str = "paper"):
    """Run the trading session with crash-safe exception handling."""
    mode = "live" if live else "paper"

    print(f"Mode: {'LIVE (real money)' if live else 'PAPER (simulated)'}")
    if live:
        print(f"Broker: {broker_type.upper()}")
    print()

    # Initialize components
    from broker import get_broker
    from risk import RiskManager
    from agent.orchestrator import TradingOrchestrator

    # Connect to broker
    print("Connecting to broker...", end=" ", flush=True)
    broker = get_broker(broker_type)
    if not broker.connect():
        print("FAILED")
        return
    print("OK")

    # Check capital before starting
    if not check_capital(broker):
        return

    # Initialize risk manager
    risk_manager = RiskManager(broker)

    # Check if we can trade
    can_trade, reason = risk_manager.can_trade()
    if not can_trade:
        print(f"\nCannot trade: {reason}")
        return

    # Show account status
    status = risk_manager.get_status()
    print()
    print(f"Portfolio:  Rs.{status['portfolio_value']:,.2f}")
    print(f"Available: Rs.{status['available_cash']:,.2f}")
    print(f"Hard Stop: Rs.{status['hard_stop']:,.2f}")
    print()

    if live:
        print("=" * 60)
        print("  WARNING: LIVE TRADING - REAL MONEY AT RISK")
        print("=" * 60)
        confirm = input("\nType 'I CONFIRM' to start live trading: ")
        if confirm != "I CONFIRM":
            print("Cancelled.")
            return
        print()

    print("Starting trading session...")
    print("Press Ctrl+C at any time to stop safely.")
    print("-" * 60)
    print()

    # Run orchestrator with crash-safe wrapper
    orchestrator = TradingOrchestrator(broker, risk_manager)
    try:
        orchestrator.run(paper_mode=(mode == "paper"))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.exception(f"Orchestrator crashed: {e}")
        from utils.alerts import alert_error
        alert_error("Orchestrator crash", str(e))
    finally:
        # Ensure positions are cleaned up even on crash
        if orchestrator.active_trades:
            logger.warning(f"Cleaning up {len(orchestrator.active_trades)} active positions after crash...")
            try:
                orchestrator._exit_all_positions("System crash cleanup")
            except Exception as e2:
                logger.error(f"Failed to clean up positions: {e2}")
                from utils.alerts import alert_error
                alert_error("CRITICAL: Position cleanup failed",
                           f"{len(orchestrator.active_trades)} positions may be orphaned: {e2}")


# ============================================
# MAIN
# ============================================

def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    raise KeyboardInterrupt  # Reuse existing KeyboardInterrupt handler


def main():
    import argparse

    # Register signal handlers for graceful shutdown (Docker, systemd, etc.)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Daily Trading Agent - Run every morning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py              Paper trading (safe, no real money)
  python start.py --live       Live trading (uses BROKER env var)
  python start.py --zerodha    Live trading with Zerodha (auto-login via TOTP)

First time setup:
  Upstox:   python scripts/upstox_auth.py
  Zerodha:  python scripts/zerodha_setup.py
  Test:     python trade.py test --live
        """
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use real money (default: paper trading)"
    )
    parser.add_argument(
        "--zerodha", action="store_true",
        help="Use Zerodha as live broker (auto-login via TOTP)"
    )

    args = parser.parse_args()

    # --zerodha implies --live
    live = args.live or args.zerodha
    broker_type = _resolve_broker_type(args) if live else "paper"

    # Step 0: Security check
    check_env_permissions()

    # Step 1: Banner
    print_banner()

    # Step 2: Acquire file lock (prevent double instances)
    lock = FileLock()
    if not lock.acquire():
        # Check if the PID in the lock file is from a dead process
        try:
            content = LOCK_FILE.read_text()
            for line in content.splitlines():
                if line.startswith("PID:"):
                    old_pid = int(line.split(":")[1].strip())
                    if not is_pid_running(old_pid):
                        logger.warning(f"Stale lock from dead PID {old_pid}, cleaning up")
                        LOCK_FILE.unlink(missing_ok=True)
                        if lock.acquire():
                            print("File lock acquired (cleaned stale lock)")
                            break
        except Exception:
            pass

        if not lock._fd:
            print("ERROR: Another trading instance is already running!")
            print(f"Lock file: {LOCK_FILE}")
            print("If this is wrong, delete the lock file and try again.")
            return
    else:
        print("File lock acquired (single instance guaranteed)")

    try:
        # Step 3: Check trading day
        if not check_trading_day():
            return

        # Step 4: Check broker token (only for live)
        if broker_type == 'zerodha':
            if not check_zerodha_token(live):
                return
        else:
            if not check_upstox_token(live):
                return

        # Step 5: Wait for market if early
        if not wait_for_market():
            return

        # Step 6: Run trading session
        try:
            run_trading_session(live, broker_type)
        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
            print(f"\nFatal error: {e}")
            print("Check logs/trading_*.log for details")

        print("\nDone. See you tomorrow!")

    finally:
        lock.release()
        print("File lock released.")


if __name__ == "__main__":
    main()
