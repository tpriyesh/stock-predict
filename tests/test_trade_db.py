"""
Tests for utils/trade_db.py - SQLite trade persistence layer.

Uses the `trade_db` fixture from conftest.py which provides a TradeDB
backed by a temporary file that is auto-cleaned after each test.
"""
import json
import sqlite3
import threading
from datetime import datetime, date, timedelta
from pathlib import Path

import pytest


# ============================================
# HELPERS
# ============================================

def _insert_trade_with_date(trade_db, trade_id: str, entry_date: str, status: str = "OPEN"):
    """
    Insert a trade row directly via SQL so we can control entry_time.
    This bypasses record_entry() which always uses datetime.now().
    """
    with trade_db._conn() as conn:
        conn.execute("""
            INSERT INTO trades (
                trade_id, symbol, side, quantity, entry_price,
                stop_loss, original_stop_loss, current_stop,
                highest_price, target, signal_confidence,
                order_ids, entry_time, status, broker_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, "TESTSTOCK", "BUY", 10, 100.0,
            95.0, 95.0, 95.0,
            100.0, 110.0, 0.7,
            json.dumps(["ORD001"]),
            entry_date,
            status, "paper"
        ))


def _insert_closed_trade_with_dates(
    trade_db, trade_id: str, entry_date: str, exit_date: str, pnl: float
):
    """Insert a CLOSED trade with controlled entry and exit timestamps."""
    with trade_db._conn() as conn:
        conn.execute("""
            INSERT INTO trades (
                trade_id, symbol, side, quantity, entry_price,
                exit_price, stop_loss, original_stop_loss, current_stop,
                highest_price, target, pnl, signal_confidence,
                order_ids, entry_time, exit_time, exit_reason,
                status, broker_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, "TESTSTOCK", "BUY", 10, 100.0,
            105.0, 95.0, 95.0, 95.0,
            105.0, 110.0, pnl, 0.7,
            json.dumps(["ORD001"]),
            entry_date, exit_date, "TARGET",
            "CLOSED", "paper"
        ))


# ============================================
# TESTS
# ============================================

class TestTradeDB:
    """Tests for the TradeDB persistence layer."""

    def test_record_entry_and_retrieve(self, trade_db):
        """record_entry persists all fields; get_trade returns them correctly."""
        trade_db.record_entry(
            trade_id="T001",
            symbol="RELIANCE",
            side="BUY",
            quantity=10,
            entry_price=2500.0,
            stop_loss=2450.0,
            target=2575.0,
            confidence=0.72,
            order_ids=["ORD001", "ORD002"],
            broker_mode="paper",
        )

        trade = trade_db.get_trade("T001")
        assert trade is not None
        assert trade["trade_id"] == "T001"
        assert trade["symbol"] == "RELIANCE"
        assert trade["side"] == "BUY"
        assert trade["quantity"] == 10
        assert trade["entry_price"] == 2500.0
        assert trade["stop_loss"] == 2450.0
        assert trade["original_stop_loss"] == 2450.0
        assert trade["current_stop"] == 2450.0
        assert trade["highest_price"] == 2500.0  # Set to entry_price on entry
        assert trade["target"] == 2575.0
        assert trade["signal_confidence"] == 0.72
        assert json.loads(trade["order_ids"]) == ["ORD001", "ORD002"]
        assert trade["status"] == "OPEN"
        assert trade["broker_mode"] == "paper"
        assert trade["exit_price"] is None
        assert trade["exit_time"] is None

    def test_record_exit_updates_status(self, trade_db):
        """record_exit sets status to CLOSED with correct pnl and exit fields."""
        trade_db.record_entry(
            trade_id="T002",
            symbol="INFY",
            side="BUY",
            quantity=5,
            entry_price=1500.0,
            stop_loss=1470.0,
            target=1545.0,
            confidence=0.65,
            order_ids=["ORD010"],
            broker_mode="paper",
        )

        trade_db.record_exit(
            trade_id="T002",
            exit_price=1540.0,
            pnl=200.0,
            exit_reason="TARGET",
        )

        trade = trade_db.get_trade("T002")
        assert trade["status"] == "CLOSED"
        assert trade["exit_price"] == 1540.0
        assert trade["pnl"] == 200.0
        assert trade["exit_reason"] == "TARGET"
        assert trade["exit_time"] is not None

    def test_update_stop_persists(self, trade_db):
        """update_stop writes new current_stop and highest_price to DB."""
        trade_db.record_entry(
            trade_id="T003",
            symbol="TCS",
            side="BUY",
            quantity=3,
            entry_price=3400.0,
            stop_loss=3350.0,
            target=3475.0,
            confidence=0.68,
            order_ids=["ORD020"],
            broker_mode="paper",
        )

        trade_db.update_stop("T003", new_stop=3380.0, highest_price=3460.0)

        trade = trade_db.get_trade("T003")
        assert trade["current_stop"] == 3380.0
        assert trade["highest_price"] == 3460.0
        # original_stop_loss should be unchanged
        assert trade["original_stop_loss"] == 3350.0

    def test_orphaned_trade_detection(self, trade_db):
        """
        mark_orphaned_trades marks OPEN trades from previous days as ORPHANED
        and returns the count.
        """
        yesterday = (date.today() - timedelta(days=1)).isoformat() + "T10:00:00"
        _insert_trade_with_date(trade_db, "T_OLD", yesterday, status="OPEN")

        # Also insert a trade from today -- should NOT be orphaned
        today_ts = date.today().isoformat() + "T10:00:00"
        _insert_trade_with_date(trade_db, "T_TODAY", today_ts, status="OPEN")

        count = trade_db.mark_orphaned_trades()
        assert count == 1

        old_trade = trade_db.get_trade("T_OLD")
        assert old_trade["status"] == "ORPHANED"
        assert old_trade["exit_reason"] == "System crash - trade left open"

        today_trade = trade_db.get_trade("T_TODAY")
        assert today_trade["status"] == "OPEN"

    def test_today_trades_filter(self, trade_db):
        """get_today_trades returns only trades with entry_time matching today."""
        yesterday = (date.today() - timedelta(days=1)).isoformat() + "T14:30:00"
        today_ts = date.today().isoformat() + "T09:30:00"

        _insert_trade_with_date(trade_db, "T_YEST", yesterday)
        _insert_trade_with_date(trade_db, "T_TOD", today_ts)

        today_trades = trade_db.get_today_trades()
        trade_ids = [t["trade_id"] for t in today_trades]
        assert "T_TOD" in trade_ids
        assert "T_YEST" not in trade_ids

    def test_today_pnl_sum(self, trade_db):
        """get_today_pnl returns the sum of pnl for today's CLOSED trades."""
        today_entry = date.today().isoformat() + "T09:30:00"
        today_exit = date.today().isoformat() + "T14:00:00"

        _insert_closed_trade_with_dates(
            trade_db, "T_W1", today_entry, today_exit, pnl=150.0
        )
        _insert_closed_trade_with_dates(
            trade_db, "T_W2", today_entry, today_exit, pnl=-50.0
        )

        total = trade_db.get_today_pnl()
        assert total == pytest.approx(100.0)

    def test_daily_summary_save_load(self, trade_db):
        """save_daily_summary persists; get_performance_history retrieves it."""
        trade_db.save_daily_summary(
            total_trades=5,
            winners=3,
            losers=2,
            total_pnl=450.0,
            portfolio_value=100450.0,
            broker_mode="paper",
        )

        history = trade_db.get_performance_history(days=7)
        assert len(history) >= 1
        today_summary = history[0]
        assert today_summary["trade_date"] == date.today().isoformat()
        assert today_summary["total_trades"] == 5
        assert today_summary["winners"] == 3
        assert today_summary["losers"] == 2
        assert today_summary["total_pnl"] == 450.0
        assert today_summary["portfolio_value"] == 100450.0
        assert today_summary["win_rate"] == pytest.approx(60.0)
        assert today_summary["broker_mode"] == "paper"

    def test_position_snapshot(self, trade_db):
        """save_position_snapshot writes rows that can be read back."""
        positions = [
            {
                "symbol": "RELIANCE",
                "quantity": 10,
                "average_price": 2500.0,
                "last_price": 2520.0,
                "pnl": 200.0,
                "source": "broker",
            },
            {
                "symbol": "INFY",
                "quantity": 5,
                "average_price": 1500.0,
                "last_price": 1490.0,
                "pnl": -50.0,
                "source": "broker",
            },
        ]

        trade_db.save_position_snapshot(positions)

        # Verify directly in DB
        with trade_db._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM position_snapshots ORDER BY id"
            ).fetchall()
            rows = [dict(r) for r in rows]

        assert len(rows) == 2
        assert rows[0]["symbol"] == "RELIANCE"
        assert rows[0]["quantity"] == 10
        assert rows[0]["average_price"] == 2500.0
        assert rows[0]["last_price"] == 2520.0
        assert rows[0]["unrealized_pnl"] == 200.0
        assert rows[1]["symbol"] == "INFY"
        assert rows[1]["quantity"] == 5

    def test_add_order_id(self, trade_db):
        """add_order_id appends a new order ID to the existing JSON list."""
        trade_db.record_entry(
            trade_id="T004",
            symbol="HDFCBANK",
            side="BUY",
            quantity=8,
            entry_price=1600.0,
            stop_loss=1570.0,
            target=1645.0,
            confidence=0.60,
            order_ids=["ORD100"],
            broker_mode="paper",
        )

        trade_db.add_order_id("T004", "ORD101")
        trade_db.add_order_id("T004", "ORD102")

        trade = trade_db.get_trade("T004")
        ids = json.loads(trade["order_ids"])
        assert ids == ["ORD100", "ORD101", "ORD102"]

    def test_concurrent_access(self, tmp_path):
        """
        Two TradeDB instances on the same file can write concurrently
        without corruption, thanks to WAL mode.
        """
        from utils.trade_db import TradeDB

        db_path = tmp_path / "concurrent_test.db"
        db1 = TradeDB(db_path=db_path)
        db2 = TradeDB(db_path=db_path)

        errors = []

        def writer(db, prefix, count):
            try:
                for i in range(count):
                    db.record_entry(
                        trade_id=f"{prefix}_{i:03d}",
                        symbol="TESTSTOCK",
                        side="BUY",
                        quantity=1,
                        entry_price=100.0,
                        stop_loss=95.0,
                        target=110.0,
                        confidence=0.5,
                        order_ids=[f"ORD_{prefix}_{i}"],
                        broker_mode="paper",
                    )
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=(db1, "A", 20))
        t2 = threading.Thread(target=writer, args=(db2, "B", 20))

        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert len(errors) == 0, f"Concurrent writes raised errors: {errors}"

        # Verify all 40 trades exist
        all_open = db1.get_open_trades()
        trade_ids = {t["trade_id"] for t in all_open}
        assert len(trade_ids) == 40
        for i in range(20):
            assert f"A_{i:03d}" in trade_ids
            assert f"B_{i:03d}" in trade_ids
