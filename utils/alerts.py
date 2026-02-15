"""
Alerts - Telegram notifications for critical trading events.

Setup:
1. Create a Telegram bot via @BotFather
2. Get your chat ID via @userinfobot
3. Set in .env:
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
"""
import os
import threading
from datetime import datetime
from typing import Optional
from loguru import logger

from utils.platform import now_ist


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Environment tag: "PROD" on Hostinger, "DEV" on local machine
ALERT_ENV = os.getenv("ALERT_ENV", "DEV")


def _send_telegram(message: str, retries: int = 1):
    """Send a Telegram message (non-blocking, with retry)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    # Prepend environment tag to every message
    tagged_message = f"[{ALERT_ENV}] {message}"

    def _do_send():
        import time as _time
        for attempt in range(retries + 1):
            try:
                import requests
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                resp = requests.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": tagged_message,
                    "parse_mode": "HTML"
                }, timeout=10)
                if resp.status_code == 200:
                    return
                logger.warning(f"Telegram HTTP {resp.status_code} (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"Telegram send failed (attempt {attempt+1}): {e}")
            if attempt < retries:
                _time.sleep(2)

    threading.Thread(target=_do_send, daemon=True).start()


def alert_trade_entry(symbol: str, quantity: int, price: float, stop_loss: float, target: float):
    """Alert on trade entry."""
    msg = (
        f"<b>BUY {symbol}</b>\n"
        f"Qty: {quantity} @ Rs.{price:.2f}\n"
        f"SL: Rs.{stop_loss:.2f} | Target: Rs.{target:.2f}\n"
        f"Time: {now_ist().strftime('%H:%M:%S')}"
    )
    _send_telegram(msg)


def alert_trade_exit(symbol: str, quantity: int, entry: float, exit_price: float, pnl: float, reason: str):
    """Alert on trade exit."""
    icon = "+" if pnl >= 0 else ""
    msg = (
        f"<b>EXIT {symbol}</b> ({reason})\n"
        f"Qty: {quantity} | Entry: Rs.{entry:.2f} | Exit: Rs.{exit_price:.2f}\n"
        f"P&L: <b>{icon}Rs.{pnl:.2f}</b>\n"
        f"Time: {now_ist().strftime('%H:%M:%S')}"
    )
    _send_telegram(msg)


def alert_kill_switch(portfolio_value: float, hard_stop: float, reason: str):
    """Alert on kill switch activation (critical - extra retry)."""
    msg = (
        f"KILL SWITCH TRIGGERED\n"
        f"Portfolio: Rs.{portfolio_value:,.0f}\n"
        f"Hard Stop: Rs.{hard_stop:,.0f}\n"
        f"Reason: {reason}\n"
        f"ALL TRADING STOPPED"
    )
    _send_telegram(msg, retries=2)


def alert_error(context: str, error: str):
    """Alert on critical error (extra retry)."""
    msg = (
        f"ERROR: {context}\n"
        f"{error[:200]}\n"
        f"Time: {now_ist().strftime('%H:%M:%S')}"
    )
    _send_telegram(msg, retries=2)


def alert_daily_report(
    total_trades: int, winners: int, total_pnl: float,
    portfolio_value: float
):
    """Send daily trading report."""
    win_rate = winners / total_trades * 100 if total_trades > 0 else 0
    icon = "+" if total_pnl >= 0 else ""
    msg = (
        f"<b>DAILY REPORT</b>\n"
        f"Trades: {total_trades} | Win Rate: {win_rate:.0f}%\n"
        f"P&L: <b>{icon}Rs.{total_pnl:.2f}</b>\n"
        f"Portfolio: Rs.{portfolio_value:,.0f}\n"
        f"Date: {now_ist().strftime('%Y-%m-%d')}"
    )
    _send_telegram(msg)


def alert_position_reconciliation_mismatch(
    broker_positions: list, internal_positions: list
):
    """Alert when broker and internal positions don't match."""
    msg = (
        f"POSITION MISMATCH\n"
        f"Broker: {broker_positions}\n"
        f"Internal: {internal_positions}\n"
        f"Manual review needed!"
    )
    _send_telegram(msg)


def alert_sl_modify_failed(symbol: str, old_stop: float, new_stop: float):
    """Alert when stop loss modification fails."""
    msg = (
        f"SL MODIFY FAILED: {symbol}\n"
        f"Wanted: Rs.{old_stop:.2f} -> Rs.{new_stop:.2f}\n"
        f"Broker SL may be stale!"
    )
    _send_telegram(msg)


def alert_startup(mode: str, portfolio_value: float):
    """Alert on system startup."""
    msg = (
        f"Trading Agent STARTED\n"
        f"Mode: {mode.upper()}\n"
        f"Portfolio: Rs.{portfolio_value:,.0f}\n"
        f"Time: {now_ist().strftime('%H:%M:%S')}"
    )
    _send_telegram(msg)
