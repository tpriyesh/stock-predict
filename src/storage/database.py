"""
DuckDB database operations for time-series storage.
"""
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd
from loguru import logger

from .models import StockPrice, NewsArticle, Prediction


class Database:
    """DuckDB database manager for stock prediction data."""

    def __init__(self, db_path: Path | str):
        """
        Initialize database connection.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()
        logger.info(f"Database initialized at {self.db_path}")

    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                adj_close DOUBLE,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                url VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                description VARCHAR,
                content VARCHAR,
                published_at TIMESTAMP NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tickers VARCHAR[],
                event_type VARCHAR,
                sentiment VARCHAR,
                sentiment_score DOUBLE,
                key_claims VARCHAR[],
                relevance_score DOUBLE
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id VARCHAR PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                generated_at TIMESTAMP NOT NULL,
                trade_type VARCHAR NOT NULL,
                signal VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                current_price DOUBLE NOT NULL,
                entry_price DOUBLE NOT NULL,
                stop_loss DOUBLE NOT NULL,
                target_price DOUBLE NOT NULL,
                summary VARCHAR,
                reasons VARCHAR[],
                outcome_pct DOUBLE,
                outcome_hit_target BOOLEAN,
                outcome_hit_stop BOOLEAN
            )
        """)

        # Create indices for faster queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(generated_at)")

    def save_prices(self, prices: list[StockPrice]) -> int:
        """
        Save price data (upsert).

        Args:
            prices: List of StockPrice objects

        Returns:
            Number of rows inserted/updated
        """
        if not prices:
            return 0

        df = pd.DataFrame([p.model_dump() for p in prices])

        # Use INSERT OR REPLACE
        self.conn.execute("""
            INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, volume, adj_close)
            SELECT * FROM df
        """)

        logger.info(f"Saved {len(prices)} price records")
        return len(prices)

    def get_prices(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get price data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with OHLCV data
        """
        query = "SELECT * FROM prices WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        return self.conn.execute(query, params).df()

    def get_latest_price_date(self, symbol: str) -> Optional[date]:
        """Get the most recent price date for a symbol."""
        result = self.conn.execute(
            "SELECT MAX(date) FROM prices WHERE symbol = ?",
            [symbol]
        ).fetchone()
        return result[0] if result and result[0] else None

    def save_news(self, articles: list[NewsArticle]) -> int:
        """Save news articles."""
        if not articles:
            return 0

        for article in articles:
            self.conn.execute("""
                INSERT OR REPLACE INTO news
                (id, source, url, title, description, content, published_at,
                 fetched_at, tickers, event_type, sentiment, sentiment_score,
                 key_claims, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                article.id or str(hash(article.url)),
                article.source,
                article.url,
                article.title,
                article.description,
                article.content,
                article.published_at,
                article.fetched_at,
                article.tickers,
                article.event_type.value,
                article.sentiment.value,
                article.sentiment_score,
                article.key_claims,
                article.relevance_score
            ])

        logger.info(f"Saved {len(articles)} news articles")
        return len(articles)

    def get_recent_news(
        self,
        symbols: Optional[list[str]] = None,
        hours: int = 72
    ) -> pd.DataFrame:
        """Get recent news articles."""
        query = f"""
            SELECT * FROM news
            WHERE published_at >= CURRENT_TIMESTAMP - INTERVAL '{hours} hours'
        """

        if symbols:
            # Filter by ticker mentions
            query += f" AND list_has_any(tickers, {symbols})"

        query += " ORDER BY published_at DESC"

        return self.conn.execute(query).df()

    def save_prediction(self, prediction: Prediction) -> None:
        """Save a prediction."""
        self.conn.execute("""
            INSERT OR REPLACE INTO predictions
            (id, symbol, generated_at, trade_type, signal, confidence,
             current_price, entry_price, stop_loss, target_price,
             summary, reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            prediction.id or f"{prediction.symbol}_{prediction.generated_at.isoformat()}",
            prediction.symbol,
            prediction.generated_at,
            prediction.trade_type.value,
            prediction.signal.value,
            prediction.confidence,
            prediction.current_price,
            prediction.entry_price,
            prediction.stop_loss,
            prediction.target_price,
            prediction.summary,
            prediction.reasons
        ])

    def get_predictions(
        self,
        date: Optional[date] = None,
        trade_type: Optional[str] = None
    ) -> pd.DataFrame:
        """Get predictions for a given date."""
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if date:
            query += " AND DATE(generated_at) = ?"
            params.append(date)
        if trade_type:
            query += " AND trade_type = ?"
            params.append(trade_type)

        query += " ORDER BY confidence DESC"

        return self.conn.execute(query, params).df()

    def update_prediction_outcome(
        self,
        prediction_id: str,
        outcome_pct: float,
        hit_target: bool,
        hit_stop: bool
    ) -> None:
        """Update prediction with actual outcome."""
        self.conn.execute("""
            UPDATE predictions
            SET outcome_pct = ?, outcome_hit_target = ?, outcome_hit_stop = ?
            WHERE id = ?
        """, [outcome_pct, hit_target, hit_stop, prediction_id])

    def get_accuracy_stats(self, days: int = 30) -> dict:
        """Get accuracy statistics for recent predictions."""
        result = self.conn.execute(f"""
            SELECT
                trade_type,
                COUNT(*) as total,
                SUM(CASE WHEN outcome_hit_target THEN 1 ELSE 0 END) as hits,
                AVG(outcome_pct) as avg_return
            FROM predictions
            WHERE generated_at >= CURRENT_TIMESTAMP - INTERVAL '{days} days'
              AND outcome_pct IS NOT NULL
            GROUP BY trade_type
        """).df()

        return result.to_dict('records')

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")
