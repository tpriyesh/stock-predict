#!/usr/bin/env python3
"""
Trading Agent CLI - Main entry point.

Usage:
    python trade.py start              # Start paper trading
    python trade.py start --live       # Start live trading (uses BROKER env var)
    python trade.py start --zerodha    # Start live trading with Zerodha
    python trade.py status             # Show current status
    python trade.py positions          # Show open positions
    python trade.py pnl                # Show today's P&L
    python trade.py test               # Test broker connection
    python trade.py test --zerodha     # Test Zerodha connection
    python trade.py signals            # Get trading signals
    python trade.py signals -s RELIANCE   # Analyze specific stock
    python trade.py history            # Show trade history from DB
    python trade.py report             # Show daily paper trading report

Broker selection (for --live mode):
    --zerodha flag overrides BROKER env var to 'zerodha'
    BROKER=zerodha in .env selects Zerodha as default live broker
    BROKER=upstox (or unset) selects Upstox as default live broker
"""
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    "logs/trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)
# Issues-only log: WARNING and above
logger.add(
    "logs/issues_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="WARNING",
    format="{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)


def _resolve_live_broker(args) -> str:
    """Determine which live broker to use based on flags and env var.

    Priority: --zerodha flag > BROKER env var > default 'upstox'
    """
    if getattr(args, 'zerodha', False):
        return 'zerodha'
    return os.getenv('BROKER', 'upstox').lower()


def _get_broker(args=None, live: bool = False):
    """Get the correct broker based on mode and flags."""
    from broker import get_broker
    if live and args is not None:
        broker_type = _resolve_live_broker(args)
    elif live:
        broker_type = os.getenv('BROKER', 'upstox').lower()
    else:
        broker_type = 'paper'
    broker = get_broker(broker_type)
    if not broker.connect():
        print(f"Failed to connect to {broker_type} broker!")
        sys.exit(1)
    return broker


def cmd_start(args):
    """Start the trading agent"""
    live = args.live or getattr(args, 'zerodha', False)
    mode = 'live' if live else 'paper'

    if mode == 'live':
        broker_name = _resolve_live_broker(args)
        print("\n" + "="*60)
        print("  LIVE TRADING MODE")
        print("="*60)
        print("You are about to start LIVE trading with REAL money.")
        print(f"Broker: {broker_name.upper()}")
        print("\nCapital at risk: Rs.1,00,000")
        print("Hard Stop: Rs.80,000")
        print("="*60)

        confirm = input("\nType 'I CONFIRM' to proceed: ")
        if confirm != 'I CONFIRM':
            print("Cancelled.")
            return

    print(f"\nStarting trading agent in {mode.upper()} mode...")
    print("="*60)

    from broker import get_broker
    from risk import RiskManager

    # Initialize broker
    broker_type = _resolve_live_broker(args) if mode == 'live' else 'paper'
    broker = get_broker(broker_type)

    if not broker.connect():
        print("Failed to connect to broker!")
        if mode == 'live':
            if broker_type == 'zerodha':
                print("   Run: python scripts/zerodha_setup.py")
            else:
                print("   Run: python scripts/upstox_auth.py")
        return

    # Initialize risk manager
    risk_manager = RiskManager(broker)

    # Check if we can trade
    can_trade, reason = risk_manager.can_trade()
    if not can_trade:
        print(f"Cannot trade: {reason}")
        return

    # Show status
    status = risk_manager.get_status()
    print(f"\nStatus:")
    print(f"   Portfolio Value: Rs.{status['portfolio_value']:,.2f}")
    print(f"   Available Cash:  Rs.{status['available_cash']:,.2f}")
    print(f"   Hard Stop:       Rs.{status['hard_stop']:,.2f}")
    print(f"   Positions:       {status['positions']}/{status['max_positions']}")

    if mode == 'paper':
        print("\n   Paper trading mode - no real orders will be placed")
        print("   Use this mode to test your strategies")

    print("\n   Agent ready. Press Ctrl+C to stop.")

    # Start trading orchestrator with crash-safe handling
    from agent.orchestrator import TradingOrchestrator

    print("\n   Initializing trading orchestrator...")
    orchestrator = TradingOrchestrator(broker, risk_manager)

    print("   Starting trading loop. Press Ctrl+C to stop gracefully.\n")
    try:
        orchestrator.run(paper_mode=(mode == 'paper'))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt - shutting down")
    except Exception as e:
        logger.exception(f"Orchestrator crashed: {e}")
        print(f"\nCRASH: {e}")
    finally:
        # Ensure cleanup even on crash
        if orchestrator.active_trades:
            print(f"\nCleaning up {len(orchestrator.active_trades)} active positions...")
            try:
                orchestrator._exit_all_positions("CLI crash cleanup")
            except Exception as cleanup_err:
                logger.error(f"Position cleanup failed: {cleanup_err}")
                print(f"WARNING: Position cleanup failed: {cleanup_err}")
                print("Check broker for open positions!")


def cmd_status(args):
    """Show current status"""
    from risk import RiskManager

    live = (args.live if hasattr(args, 'live') else False) or getattr(args, 'zerodha', False)
    broker = _get_broker(args, live)
    risk_manager = RiskManager(broker)

    status = risk_manager.get_status()

    print("\n" + "="*60)
    print("TRADING AGENT STATUS")
    print("="*60)

    # Trading status
    if status['can_trade']:
        print("Status: READY TO TRADE")
    else:
        print(f"Status: BLOCKED - {status['reason']}")

    if status['is_killed']:
        print("KILL SWITCH: ACTIVE")

    print(f"\nCapital:")
    print(f"   Portfolio:  Rs.{status['portfolio_value']:,.2f}")
    print(f"   Available:  Rs.{status['available_cash']:,.2f}")
    print(f"   Hard Stop:  Rs.{status['hard_stop']:,.2f}")
    print(f"   Buffer:     Rs.{status['distance_to_stop']:,.2f}")

    print(f"\nToday:")
    print(f"   P&L:        Rs.{status['daily_pnl']:,.2f}")
    print(f"   Trades:     {status['trades_today']}")
    print(f"   Max Loss:   -Rs.{status['max_daily_loss']:,.2f}")

    print(f"\nPositions: {status['positions']}/{status['max_positions']}")


def cmd_positions(args):
    """Show open positions"""
    live = (args.live if hasattr(args, 'live') else False) or getattr(args, 'zerodha', False)
    broker = _get_broker(args, live)

    positions = broker.get_positions()

    print("\n" + "="*60)
    print("OPEN POSITIONS")
    print("="*60)

    if not positions:
        print("No open positions")
        return

    total_pnl = 0
    for pos in positions:
        pnl_icon = "+" if pos.pnl >= 0 else ""
        print(f"\n{pos.symbol}:")
        print(f"   Qty: {pos.quantity} @ Rs.{pos.average_price:.2f}")
        print(f"   LTP: Rs.{pos.last_price:.2f}")
        print(f"   P&L: {pnl_icon}Rs.{pos.pnl:,.2f} ({pos.pnl_pct:+.2f}%)")
        total_pnl += pos.pnl

    print(f"\n{'='*60}")
    pnl_icon = "+" if total_pnl >= 0 else ""
    print(f"Total Unrealized P&L: {pnl_icon}Rs.{total_pnl:,.2f}")


def cmd_pnl(args):
    """Show P&L"""
    live = (args.live if hasattr(args, 'live') else False) or getattr(args, 'zerodha', False)
    broker = _get_broker(args, live)

    pnl = broker.get_pnl()
    funds = broker.get_funds()

    print("\n" + "="*60)
    print("PROFIT & LOSS")
    print("="*60)

    print(f"\nToday's P&L:")
    print(f"   Realized:   Rs.{pnl.realized:,.2f}")
    print(f"   Unrealized: Rs.{pnl.unrealized:,.2f}")

    pnl_icon = "+" if pnl.total >= 0 else ""
    print(f"   Total: {pnl_icon}Rs.{pnl.total:,.2f}")

    pnl_pct = (pnl.total / funds.total_balance) * 100 if funds.total_balance > 0 else 0
    print(f"   Return:     {pnl_pct:+.2f}%")

    # Show DB history
    try:
        from utils.trade_db import get_trade_db
        db = get_trade_db()
        today_trades = db.get_today_trades()
        if today_trades:
            print(f"\nDB Trades Today: {len(today_trades)}")
            for t in today_trades:
                status = t['status']
                pnl_val = t.get('pnl', 0) or 0
                icon = "+" if pnl_val >= 0 else ""
                print(f"   {t['symbol']}: {t['side']} {t['quantity']} @ Rs.{t['entry_price']:.2f} "
                      f"[{status}] P&L={icon}Rs.{pnl_val:.2f}")
    except Exception:
        pass


def cmd_test(args):
    """Test broker connection"""
    live = args.live or getattr(args, 'zerodha', False)
    if live:
        broker_type = _resolve_live_broker(args)
    else:
        broker_type = 'paper'
    print(f"\nTesting {broker_type} broker connection...")

    from broker import get_broker

    broker = get_broker(broker_type)

    # Test connection
    if broker.connect():
        print("Connection: OK")
    else:
        print("Connection: FAILED")
        if broker_type == 'zerodha':
            print("   Run: python scripts/zerodha_auto_login.py")
        elif broker_type == 'upstox':
            print("   Run: python scripts/upstox_auth.py")
        return

    # Test market data
    print("\nTesting market data...")
    try:
        ltp = broker.get_ltp('RELIANCE')
        print(f"   RELIANCE LTP: Rs.{ltp:.2f}")
        print("Market data: OK")
    except Exception as e:
        print(f"Market data: FAILED - {e}")

    # Test funds
    print("\nTesting account...")
    try:
        funds = broker.get_funds()
        print(f"   Available: Rs.{funds.available_cash:,.2f}")
        print("Account: OK")
    except Exception as e:
        print(f"Account: FAILED - {e}")

    # Test DB
    print("\nTesting trade database...")
    try:
        from utils.trade_db import get_trade_db
        db = get_trade_db()
        open_trades = db.get_open_trades()
        print(f"   DB: OK (path={db.db_path})")
        if open_trades:
            print(f"   WARNING: {len(open_trades)} open trades in DB")
    except Exception as e:
        print(f"Database: FAILED - {e}")

    print("\nAll tests passed!")


def cmd_supertrend(args):
    """Test Supertrend indicator"""
    from features.supertrend import demo_supertrend
    demo_supertrend()


def cmd_signals(args):
    """Get current trading signals from stock-predict"""
    print("\nGenerating trading signals...")
    print("=" * 60)

    from agent.signal_adapter import SignalAdapter, TradeDecision

    adapter = SignalAdapter()

    # Get specific symbol or all
    if args.symbol:
        print(f"\nAnalyzing {args.symbol}...")
        signal = adapter.get_single_signal(args.symbol)
        if signal:
            _print_signal(signal)
        else:
            print(f"No actionable signal for {args.symbol}")
    else:
        print("\nAnalyzing universe...")
        signals = adapter.get_trade_signals()

        if not signals:
            print("No actionable signals found")
            return

        print(f"\nFound {len(signals)} actionable signals:\n")

        for i, signal in enumerate(signals[:10], 1):
            _print_signal(signal, index=i)
            print()


def cmd_history(args):
    """Show trade history from database"""
    from utils.trade_db import get_trade_db
    db = get_trade_db()

    days = args.days if hasattr(args, 'days') and args.days else 7

    print("\n" + "="*60)
    print("TRADE HISTORY")
    print("="*60)

    history = db.get_performance_history(days)
    if not history:
        print("No trading history found")
        return

    total_pnl = 0
    for day in reversed(history):
        pnl = day['total_pnl'] or 0
        total_pnl += pnl
        icon = "+" if pnl >= 0 else ""
        wr = day.get('win_rate', 0) or 0
        print(
            f"  {day['trade_date']}: {day['total_trades']} trades, "
            f"WR={wr:.0f}%, P&L={icon}Rs.{pnl:.2f}, "
            f"Portfolio=Rs.{day.get('portfolio_value', 0):,.0f}"
        )

    print(f"\n  Total P&L ({len(history)} days): {'+'if total_pnl>=0 else ''}Rs.{total_pnl:.2f}")


def cmd_report(args):
    """Show daily paper trading report with daemon status."""
    from utils.trade_db import get_trade_db
    from utils.platform import today_ist, now_ist, is_pid_running

    db = get_trade_db()
    today = today_ist()
    today_str = today.isoformat()

    print("\n" + "="*60)
    print("  PAPER TRADING DAILY REPORT")
    print(f"  {today.strftime('%A, %B %d, %Y')}")
    print("="*60)

    # Daemon status
    pid_file = Path.home() / ".trading_daemon.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if is_pid_running(pid):
                print(f"\n  Daemon: RUNNING (PID: {pid})")
            else:
                print("\n  Daemon: NOT RUNNING (stale PID file)")
        except (ValueError, OSError):
            print("\n  Daemon: UNKNOWN")
    else:
        print("\n  Daemon: NOT RUNNING")
        print("  Start with: bash scripts/setup_macos.sh install")

    # Today's summary from DB
    print("\n" + "-"*60)
    print("  TODAY'S SUMMARY")
    print("-"*60)

    today_trades = db.get_today_trades()
    history = db.get_performance_history(1)
    today_summary = None
    if history and history[0].get('trade_date') == today_str:
        today_summary = history[0]

    if today_summary:
        pnl = today_summary['total_pnl'] or 0
        icon = "+" if pnl >= 0 else ""
        wr = today_summary.get('win_rate', 0) or 0
        print(f"  Trades:     {today_summary['total_trades']}")
        print(f"  Winners:    {today_summary.get('winners', 0)}")
        print(f"  Losers:     {today_summary.get('losers', 0)}")
        print(f"  Win Rate:   {wr:.0f}%")
        print(f"  P&L:        {icon}Rs.{pnl:.2f}")
        pv = today_summary.get('portfolio_value', 0)
        if pv:
            print(f"  Portfolio:  Rs.{pv:,.2f}")
    elif today_trades:
        total_pnl = sum((t.get('pnl', 0) or 0) for t in today_trades)
        winners = sum(1 for t in today_trades if (t.get('pnl', 0) or 0) > 0)
        losers = sum(1 for t in today_trades if (t.get('pnl', 0) or 0) < 0)
        icon = "+" if total_pnl >= 0 else ""
        print(f"  Trades:     {len(today_trades)}")
        print(f"  Winners:    {winners}")
        print(f"  Losers:     {losers}")
        print(f"  P&L:        {icon}Rs.{total_pnl:.2f}")
    else:
        print("  No trades today")

    # Per-trade breakdown
    if today_trades:
        print("\n" + "-"*60)
        print("  TRADE DETAILS")
        print("-"*60)
        for t in today_trades:
            pnl_val = t.get('pnl', 0) or 0
            icon = "+" if pnl_val >= 0 else ""
            status = t.get('status', '?')
            entry = t.get('entry_price', 0) or 0
            exit_p = t.get('exit_price', 0)
            exit_str = f" -> Rs.{exit_p:.2f}" if exit_p else ""
            print(
                f"  {t['symbol']}: {t.get('side', '?')} {t.get('quantity', 0)} "
                f"@ Rs.{entry:.2f}{exit_str} [{status}] "
                f"P&L={icon}Rs.{pnl_val:.2f}"
            )

    # Weekly performance
    print("\n" + "-"*60)
    print("  WEEKLY PERFORMANCE (last 5 trading days)")
    print("-"*60)

    weekly = db.get_performance_history(5)
    if weekly:
        week_pnl = 0
        for day in reversed(weekly):
            pnl = day['total_pnl'] or 0
            week_pnl += pnl
            icon = "+" if pnl >= 0 else ""
            wr = day.get('win_rate', 0) or 0
            print(
                f"  {day['trade_date']}: {day['total_trades']} trades, "
                f"WR={wr:.0f}%, P&L={icon}Rs.{pnl:.2f}"
            )
        print(f"\n  Week Total: {'+'if week_pnl>=0 else ''}Rs.{week_pnl:.2f}")
    else:
        print("  No history yet")

    # Report file
    report_file = Path("data/reports") / f"{today_str}.txt"
    if report_file.exists():
        print(f"\n  Report file: {report_file}")

    # Log file
    log_file = Path("logs") / f"daemon_{today.strftime('%Y-%m-%d')}.log"
    if log_file.exists():
        print(f"  Log file:    {log_file}")

    print()


def _print_signal(signal, index=None):
    """Print a single signal nicely"""
    from agent.signal_adapter import TradeDecision

    prefix = f"{index}. " if index else ""
    icon = "BUY" if signal.decision == TradeDecision.BUY else "SELL" if signal.decision == TradeDecision.SELL else "HOLD"

    print(f"{prefix}[{icon}] {signal.symbol}: {signal.decision.value}")
    print(f"   Confidence: {signal.confidence:.1%}")
    print(f"   Entry: Rs.{signal.entry_price:.2f}")
    print(f"   Stop Loss: Rs.{signal.stop_loss:.2f} ({signal.risk_pct:.1f}% risk)")
    print(f"   Target: Rs.{signal.target_price:.2f} ({signal.reward_pct:.1f}% reward)")
    print(f"   R:R Ratio: 1:{signal.risk_reward_ratio:.1f}")
    print(f"   Position Size: {signal.position_size_pct:.1%}")

    if signal.reasons:
        print(f"   Reasons: {signal.reasons[0][:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Trading Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trade.py start              Start paper trading
  python trade.py start --live       Start live trading (uses BROKER env var)
  python trade.py start --zerodha    Start live trading with Zerodha
  python trade.py status --live      Show status (live)
  python trade.py test --zerodha     Test Zerodha connection
  python trade.py positions          Show positions
  python trade.py pnl                Show P&L
  python trade.py signals            Get trading signals
  python trade.py signals -s RELIANCE   Analyze specific stock
  python trade.py history            Show trade history
  python trade.py report             Daily paper trading report
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Helper to add common broker flags to a subparser
    def _add_broker_flags(sub):
        sub.add_argument('--live', action='store_true', help='Use live trading (real money)')
        sub.add_argument('--zerodha', action='store_true', help='Use Zerodha as live broker')

    # start command
    start_parser = subparsers.add_parser('start', help='Start trading agent')
    _add_broker_flags(start_parser)

    # status command
    status_parser = subparsers.add_parser('status', help='Show current status')
    _add_broker_flags(status_parser)

    # positions command
    pos_parser = subparsers.add_parser('positions', help='Show open positions')
    _add_broker_flags(pos_parser)

    # pnl command
    pnl_parser = subparsers.add_parser('pnl', help='Show P&L')
    _add_broker_flags(pnl_parser)

    # test command
    test_parser = subparsers.add_parser('test', help='Test broker connection')
    _add_broker_flags(test_parser)

    # supertrend command
    subparsers.add_parser('supertrend', help='Test Supertrend indicator')

    # signals command
    signals_parser = subparsers.add_parser('signals', help='Get trading signals')
    signals_parser.add_argument('--symbol', '-s', help='Specific symbol to analyze')

    # history command
    history_parser = subparsers.add_parser('history', help='Show trade history')
    history_parser.add_argument('--days', '-d', type=int, default=7, help='Number of days')

    # report command
    subparsers.add_parser('report', help='Show daily paper trading report')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Route to command handler
    commands = {
        'start': cmd_start,
        'status': cmd_status,
        'positions': cmd_positions,
        'pnl': cmd_pnl,
        'test': cmd_test,
        'supertrend': cmd_supertrend,
        'signals': cmd_signals,
        'history': cmd_history,
        'report': cmd_report,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
