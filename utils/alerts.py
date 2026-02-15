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
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
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


@dataclass
class TradeCharges:
    """Breakdown of all charges for a single trade (buy + sell combined)."""
    brokerage: float = 0.0       # Rs.20/order x2 or 0.03% whichever lower
    stt: float = 0.0             # 0.025% on sell-side turnover (intraday equity)
    exchange_charges: float = 0.0  # NSE: 0.00345% on total turnover
    gst: float = 0.0             # 18% on (brokerage + exchange charges)
    sebi_charges: float = 0.0    # Rs.10 per crore of turnover
    stamp_duty: float = 0.0      # 0.003% on buy-side turnover

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange_charges
                + self.gst + self.sebi_charges + self.stamp_duty)


def calculate_zerodha_charges(
    buy_price: float, sell_price: float, quantity: int
) -> TradeCharges:
    """
    Calculate exact Zerodha intraday equity charges.

    Rates as of 2026 (Zerodha brokerage schedule):
    - Brokerage: Rs.20/order or 0.03% whichever is lower, per order (buy + sell)
    - STT: 0.025% on sell-side turnover (intraday equity)
    - Exchange (NSE): 0.00345% on total turnover
    - GST: 18% on (brokerage + exchange charges)
    - SEBI: Rs.10 per crore of turnover
    - Stamp duty: 0.003% on buy-side turnover
    """
    if quantity <= 0 or buy_price <= 0 or sell_price <= 0:
        return TradeCharges()

    buy_turnover = buy_price * quantity
    sell_turnover = sell_price * quantity
    total_turnover = buy_turnover + sell_turnover

    # Brokerage: min(Rs.20, 0.03% of turnover) per order, applied on both buy & sell
    brokerage_buy = min(20.0, buy_turnover * 0.0003)
    brokerage_sell = min(20.0, sell_turnover * 0.0003)
    brokerage = round(brokerage_buy + brokerage_sell, 2)

    # STT: 0.025% on sell-side only (intraday equity)
    stt = round(sell_turnover * 0.00025, 2)

    # Exchange transaction charges (NSE): 0.00345% on total turnover
    exchange_charges = round(total_turnover * 0.0000345, 2)

    # GST: 18% on (brokerage + exchange charges)
    gst = round((brokerage + exchange_charges) * 0.18, 2)

    # SEBI charges: Rs.10 per crore of turnover
    sebi_charges = round(total_turnover * 10 / 10000000, 2)

    # Stamp duty: 0.003% on buy-side turnover only
    stamp_duty = round(buy_turnover * 0.00003, 2)

    return TradeCharges(
        brokerage=brokerage,
        stt=stt,
        exchange_charges=exchange_charges,
        gst=gst,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
    )


def alert_daily_report(
    total_trades: int, winners: int, total_pnl: float,
    portfolio_value: float, trades: Optional[List] = None
):
    """
    Send detailed daily trading report with per-trade breakdown,
    brokerage charges, and tax estimation.

    Args:
        total_trades: Number of completed trades
        winners: Number of winning trades
        total_pnl: Sum of P&L across all trades (after estimated fees)
        portfolio_value: Current portfolio value
        trades: List of TradeRecord objects for detailed breakdown
    """
    win_rate = winners / total_trades * 100 if total_trades > 0 else 0
    date_str = now_ist().strftime('%Y-%m-%d')

    # --- Header ---
    lines = [
        f"<b>DAILY REPORT - {date_str}</b>",
        f"Trades: {total_trades} | Win Rate: {win_rate:.0f}%",
        "",
    ]

    # --- Per-trade breakdown ---
    total_gross_pnl = 0.0
    total_charges = TradeCharges()

    if trades:
        lines.append("<b>-- TRADES --</b>")
        for t in trades:
            entry_p = getattr(t, 'entry_price', 0.0)
            exit_p = getattr(t, 'exit_price', 0.0) or 0.0
            qty = getattr(t, 'quantity', 0)
            reason = getattr(t, 'exit_reason', '')

            gross = (exit_p - entry_p) * qty
            total_gross_pnl += gross

            charges = calculate_zerodha_charges(entry_p, exit_p, qty)
            total_charges.brokerage += charges.brokerage
            total_charges.stt += charges.stt
            total_charges.exchange_charges += charges.exchange_charges
            total_charges.gst += charges.gst
            total_charges.sebi_charges += charges.sebi_charges
            total_charges.stamp_duty += charges.stamp_duty

            net = gross - charges.total
            icon = "+" if net >= 0 else ""
            symbol = getattr(t, 'symbol', '???')
            lines.append(
                f"{symbol}: {qty}x Rs.{entry_p:.2f}→{exit_p:.2f} "
                f"= {icon}Rs.{net:.2f} ({reason})"
            )
        lines.append("")
    else:
        # Fallback: no trade list provided, use total_pnl directly
        total_gross_pnl = total_pnl

    # --- P&L Waterfall ---
    lines.append("<b>-- P&amp;L BREAKDOWN --</b>")

    gross_icon = "+" if total_gross_pnl >= 0 else ""
    lines.append(f"Gross P&amp;L: {gross_icon}Rs.{total_gross_pnl:.2f}")

    if trades:
        total_charges_val = round(total_charges.total, 2)
        lines.append(f"  Brokerage: -Rs.{total_charges.brokerage:.2f}")
        lines.append(f"  STT: -Rs.{total_charges.stt:.2f}")
        lines.append(f"  Exchange: -Rs.{total_charges.exchange_charges:.2f}")
        lines.append(f"  GST: -Rs.{total_charges.gst:.2f}")
        lines.append(f"  SEBI: -Rs.{total_charges.sebi_charges:.2f}")
        lines.append(f"  Stamp Duty: -Rs.{total_charges.stamp_duty:.2f}")
        lines.append(f"  <b>Total Charges: -Rs.{total_charges_val:.2f}</b>")
        lines.append("")

        net_profit = total_gross_pnl - total_charges_val
        net_icon = "+" if net_profit >= 0 else ""
        lines.append(f"<b>Net Profit: {net_icon}Rs.{net_profit:.2f}</b>")

        # Tax estimation (20% STCG on intraday profits, only if profitable)
        if net_profit > 0:
            tax = round(net_profit * 0.20, 2)
            take_home = round(net_profit - tax, 2)
            lines.append("")
            lines.append(f"<b>-- TAX (Estimated) --</b>")
            lines.append(f"Intraday STCG @20%: -Rs.{tax:.2f}")
            lines.append(f"<b>Take-Home: +Rs.{take_home:.2f}</b>")
        elif net_profit < 0:
            lines.append("")
            lines.append("Tax: Rs.0 (loss day, can offset future gains)")
            lines.append(f"<b>Net Loss: Rs.{net_profit:.2f}</b>")
    else:
        # No trade details available, just show total
        pnl_icon = "+" if total_pnl >= 0 else ""
        lines.append(f"<b>Net P&amp;L: {pnl_icon}Rs.{total_pnl:.2f}</b>")

    # --- Footer ---
    lines.append("")
    lines.append(f"Portfolio: Rs.{portfolio_value:,.0f}")

    msg = "\n".join(lines)
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
