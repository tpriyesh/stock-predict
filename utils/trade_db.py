"""
Trade Database - SQLite persistence for trades, orders, and position snapshots.

Provides crash recovery, audit trail, and tax compliance.
All trades are persisted immediately on entry/exit.
"""
import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from loguru import logger

from utils.platform import now_ist, today_ist, check_disk_space


DB_PATH = Path(__file__).parent.parent / "data" / "trades.db"

# Current schema version
SCHEMA_VERSION = 2

# Migration scripts (version -> SQL)
MIGRATIONS = {
    2: """
        -- v2: Add schema_version tracking + volume/corporate action columns
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        -- Add columns if they don't exist (SQLite ALTER TABLE is limited)
        -- These are safe to run multiple times due to IF NOT EXISTS
    """,
}


class TradeDB:
    """
    SQLite-backed trade persistence.

    Stores:
    - trades: entry/exit records with full audit trail
    - orders: all orders placed with status tracking
    - position_snapshots: periodic broker state for reconciliation
    - daily_summary: end-of-day P&L records
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._run_migrations()
        # Disk space check
        if not check_disk_space(self.db_path.parent, min_mb=50):
            logger.critical(f"LOW DISK SPACE: Less than 50MB free at {self.db_path.parent}")
            try:
                from utils.alerts import alert_error
                alert_error("LOW DISK SPACE",
                            f"Less than 50MB free at {self.db_path.parent}. "
                            "DB writes may fail. Free up space immediately!")
            except Exception:
                pass

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    original_stop_loss REAL,
                    current_stop REAL,
                    highest_price REAL,
                    target REAL,
                    pnl REAL DEFAULT 0,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    signal_confidence REAL,
                    order_ids TEXT,
                    status TEXT DEFAULT 'OPEN',
                    broker_mode TEXT DEFAULT 'paper',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    trade_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL,
                    trigger_price REAL,
                    fill_price REAL,
                    filled_quantity INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    tag TEXT,
                    placed_at TEXT NOT NULL,
                    updated_at TEXT,
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                );

                CREATE TABLE IF NOT EXISTS position_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    average_price REAL NOT NULL,
                    last_price REAL,
                    unrealized_pnl REAL,
                    source TEXT DEFAULT 'broker'
                );

                CREATE TABLE IF NOT EXISTS daily_summary (
                    trade_date TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    winners INTEGER DEFAULT 0,
                    losers INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    portfolio_value REAL,
                    broker_mode TEXT DEFAULT 'paper'
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
                CREATE INDEX IF NOT EXISTS idx_orders_trade_id ON orders(trade_id);
                CREATE INDEX IF NOT EXISTS idx_snapshots_time ON position_snapshots(snapshot_time);
            """)
        logger.debug(f"TradeDB initialized at {self.db_path}")

    def _run_migrations(self):
        """Run forward-only schema migrations."""
        with self._conn() as conn:
            # Create schema_version table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)

            # Get current version
            row = conn.execute(
                "SELECT MAX(version) as v FROM schema_version"
            ).fetchone()
            current_version = row['v'] if row and row['v'] else 1

            # Apply pending migrations
            for version in sorted(MIGRATIONS.keys()):
                if version > current_version:
                    logger.info(f"[DB] Applying migration v{version}...")
                    try:
                        conn.executescript(MIGRATIONS[version])
                        conn.execute(
                            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                            (version, now_ist().replace(tzinfo=None).isoformat())
                        )
                        logger.info(f"[DB] Migration v{version} applied successfully")
                    except Exception as e:
                        logger.error(f"[DB] Migration v{version} failed: {e}")
                        raise

    def get_schema_version(self) -> int:
        """Get current schema version."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT MAX(version) as v FROM schema_version"
                ).fetchone()
                return row['v'] if row and row['v'] else 1
        except Exception:
            return 1

    # === TRADE OPERATIONS ===

    def record_entry(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        order_ids: List[str],
        broker_mode: str = "paper"
    ):
        """Record a new trade entry. Called immediately after order fill."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, quantity, entry_price,
                    stop_loss, original_stop_loss, current_stop,
                    highest_price, target, signal_confidence,
                    order_ids, entry_time, status, broker_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
            """, (
                trade_id, symbol, side, quantity, entry_price,
                stop_loss, stop_loss, stop_loss,
                entry_price, target, confidence,
                json.dumps(order_ids),
                now_ist().replace(tzinfo=None).isoformat(),
                broker_mode
            ))
        logger.info(f"[DB] Trade entry recorded: {trade_id} {side} {quantity} {symbol} @ {entry_price}")

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        exit_reason: str
    ):
        """Record trade exit. Called immediately after exit order fill."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE trades SET
                    exit_price = ?,
                    pnl = ?,
                    exit_time = ?,
                    exit_reason = ?,
                    status = 'CLOSED'
                WHERE trade_id = ?
            """, (
                exit_price, pnl,
                now_ist().replace(tzinfo=None).isoformat(),
                exit_reason, trade_id
            ))
        logger.info(f"[DB] Trade exit recorded: {trade_id} @ {exit_price} P&L={pnl:+.2f} ({exit_reason})")

    def update_stop(self, trade_id: str, new_stop: float, highest_price: float):
        """Update trailing stop for an active trade."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE trades SET
                    current_stop = ?,
                    highest_price = ?
                WHERE trade_id = ? AND status = 'OPEN'
            """, (new_stop, highest_price, trade_id))

    def add_order_id(self, trade_id: str, order_id: str):
        """Add an order ID to a trade's order list."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT order_ids FROM trades WHERE trade_id = ?",
                (trade_id,)
            ).fetchone()
            if row:
                ids = json.loads(row['order_ids'] or '[]')
                ids.append(order_id)
                conn.execute(
                    "UPDATE trades SET order_ids = ? WHERE trade_id = ?",
                    (json.dumps(ids), trade_id)
                )

    # === ORDER OPERATIONS ===

    def record_order(
        self,
        order_id: str,
        trade_id: Optional[str],
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: Optional[float],
        trigger_price: Optional[float],
        status: str,
        tag: Optional[str] = None
    ):
        """Record an order placement."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders (
                    order_id, trade_id, symbol, side, order_type,
                    quantity, price, trigger_price, status, tag, placed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id, trade_id, symbol, side, order_type,
                quantity, price, trigger_price, status, tag,
                now_ist().replace(tzinfo=None).isoformat()
            ))

    def update_order_status(
        self,
        order_id: str,
        status: str,
        fill_price: Optional[float] = None,
        filled_quantity: Optional[int] = None
    ):
        """Update order status after fill/cancel/reject."""
        with self._conn() as conn:
            updates = ["status = ?", "updated_at = ?"]
            params = [status, now_ist().replace(tzinfo=None).isoformat()]

            if fill_price is not None:
                updates.append("fill_price = ?")
                params.append(fill_price)
            if filled_quantity is not None:
                updates.append("filled_quantity = ?")
                params.append(filled_quantity)

            params.append(order_id)
            conn.execute(
                f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?",
                params
            )

    # === POSITION SNAPSHOTS ===

    def save_position_snapshot(self, positions: List[Dict[str, Any]]):
        """Save current broker positions for reconciliation."""
        now = now_ist().replace(tzinfo=None).isoformat()
        with self._conn() as conn:
            for pos in positions:
                conn.execute("""
                    INSERT INTO position_snapshots (
                        snapshot_time, symbol, quantity, average_price,
                        last_price, unrealized_pnl, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    now, pos['symbol'], pos['quantity'],
                    pos['average_price'], pos.get('last_price', 0),
                    pos.get('pnl', 0), pos.get('source', 'broker')
                ))

    # === QUERIES ===

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all trades with status OPEN (for crash recovery)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_today_trades(self) -> List[Dict[str, Any]]:
        """Get all trades for today."""
        today = today_ist().isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE entry_time LIKE ? ORDER BY entry_time",
                (f"{today}%",)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_today_pnl(self) -> float:
        """Get total P&L for today's closed trades."""
        today = today_ist().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE status = 'CLOSED' AND exit_time LIKE ?",
                (f"{today}%",)
            ).fetchone()
            return row['total'] if row else 0.0

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trade by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,)
            ).fetchone()
            return dict(row) if row else None

    # === DAILY SUMMARY ===

    def save_daily_summary(
        self,
        total_trades: int,
        winners: int,
        losers: int,
        total_pnl: float,
        portfolio_value: float,
        broker_mode: str = "paper"
    ):
        """Save end-of-day summary."""
        today = today_ist().isoformat()
        win_rate = winners / total_trades * 100 if total_trades > 0 else 0
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summary (
                    trade_date, total_trades, winners, losers,
                    total_pnl, win_rate, portfolio_value, broker_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today, total_trades, winners, losers,
                total_pnl, win_rate, portfolio_value, broker_mode
            ))
        logger.info(f"[DB] Daily summary saved: {total_trades} trades, P&L={total_pnl:+.2f}")

    def get_performance_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily performance history."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_summary ORDER BY trade_date DESC LIMIT ?",
                (days,)
            ).fetchall()
            return [dict(r) for r in rows]

    # === CLEANUP ===

    def mark_orphaned_trades(self):
        """Mark trades that were OPEN but never closed (crash recovery)."""
        with self._conn() as conn:
            # Find OPEN trades from previous days
            today = today_ist().isoformat()
            conn.execute("""
                UPDATE trades SET
                    status = 'ORPHANED',
                    exit_reason = 'System crash - trade left open'
                WHERE status = 'OPEN' AND entry_time < ?
            """, (today,))
            affected = conn.execute("SELECT changes()").fetchone()[0]
            if affected > 0:
                logger.warning(f"[DB] Found {affected} orphaned trades from previous sessions")
            return affected


# Singleton
_instance: Optional[TradeDB] = None


def get_trade_db() -> TradeDB:
    """Get or create the singleton TradeDB instance."""
    global _instance
    if _instance is None:
        _instance = TradeDB()
    return _instance
