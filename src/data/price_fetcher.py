"""
Price data fetcher with robust validation.
Uses yfinance for NSE/BSE data with Yahoo Finance suffix.
Rate-limited via providers/quota.py to prevent API abuse.
"""
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from src.storage.models import StockPrice
from src.data.validators import DataValidator, DataValidationError
from src.utils.logger import audit_log
from utils.platform import yfinance_with_timeout
from providers.quota import get_quota_manager
from providers.cache import get_response_cache


class PriceFetcher:
    """
    Fetches and validates stock price data from Yahoo Finance.

    For NSE stocks, use symbol with .NS suffix (e.g., RELIANCE.NS)
    For BSE stocks, use symbol with .BO suffix (e.g., RELIANCE.BO)
    """

    # Yahoo Finance suffixes for Indian exchanges
    NSE_SUFFIX = ".NS"
    BSE_SUFFIX = ".BO"

    def __init__(self, exchange: str = "NSE"):
        """
        Initialize price fetcher.

        Args:
            exchange: "NSE" or "BSE"
        """
        self.exchange = exchange
        self.suffix = self.NSE_SUFFIX if exchange == "NSE" else self.BSE_SUFFIX

        # Register yfinance with quota manager for rate limiting
        import os
        self._quota = get_quota_manager()
        self._cache = get_response_cache()
        self._price_cache_ttl = int(os.getenv("PRICE_CACHE_TTL", "300"))
        self._quota.register(
            "yfinance",
            daily_limit=None,  # No daily limit, but enforce RPM/RPS
            rpm=int(os.getenv("YFINANCE_RPM", "60")),
            rps=int(os.getenv("YFINANCE_RPS", "2")),
        )

    def _to_yahoo_symbol(self, symbol: str) -> str:
        """Convert local symbol to Yahoo Finance symbol."""
        # Handle special characters in Indian stock symbols
        symbol = symbol.replace("&", "%26")

        if not symbol.endswith(self.suffix):
            return f"{symbol}{self.suffix}"
        return symbol

    def _from_yahoo_symbol(self, yahoo_symbol: str) -> str:
        """Convert Yahoo Finance symbol back to local symbol."""
        for suffix in [self.NSE_SUFFIX, self.BSE_SUFFIX]:
            if yahoo_symbol.endswith(suffix):
                return yahoo_symbol[:-len(suffix)].replace("%26", "&")
        return yahoo_symbol

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def fetch_prices(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        period: str = "1y"
    ) -> list[StockPrice]:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Stock symbol (without exchange suffix)
            start_date: Start date (optional if period is specified)
            end_date: End date (default: today)
            period: Fallback period if dates not specified (e.g., "1y", "6mo", "1mo")

        Returns:
            List of validated StockPrice objects

        Raises:
            DataValidationError: If data fails validation
        """
        yahoo_symbol = self._to_yahoo_symbol(symbol)

        audit_log(
            "FETCH_PRICES_START",
            symbol=symbol,
            yahoo_symbol=yahoo_symbol,
            start=start_date,
            end=end_date,
            period=period
        )

        try:
            # Check cache first
            cache_key = f"history:{symbol}:{period}:{start_date}:{end_date}"
            cached = self._cache.get("yfinance", cache_key)
            if cached is not None:
                return cached

            # Rate limit before yfinance call
            self._quota.wait_and_record("yfinance")

            _start = start_date
            _end = end_date
            _period = period

            def _fetch_history():
                ticker = yf.Ticker(yahoo_symbol)
                if _start and _end:
                    return ticker.history(
                        start=_start.isoformat(),
                        end=(_end + timedelta(days=1)).isoformat()
                    )
                else:
                    return ticker.history(period=_period)

            df = yfinance_with_timeout(_fetch_history, timeout_seconds=30)

            if df.empty:
                logger.warning(f"No data returned for {yahoo_symbol}")
                self._quota.record_success("yfinance")
                return []

            # Clean and validate
            prices = self._process_dataframe(df, symbol)

            self._quota.record_success("yfinance")

            # Cache results
            if prices:
                self._cache.set("yfinance", cache_key, prices,
                                ttl_seconds=self._price_cache_ttl)

            audit_log(
                "FETCH_PRICES_SUCCESS",
                symbol=symbol,
                rows=len(prices),
                date_range=f"{prices[0].date} to {prices[-1].date}" if prices else "N/A"
            )

            return prices

        except Exception as e:
            self._quota.record_failure("yfinance")
            logger.error(f"Failed to fetch {yahoo_symbol}: {e}")
            audit_log("FETCH_PRICES_FAILED", symbol=symbol, error=str(e))
            raise

    def _process_dataframe(self, df: pd.DataFrame, symbol: str) -> list[StockPrice]:
        """Process and validate raw DataFrame from yfinance."""

        # Rename columns to our standard
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        })

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing columns: {missing}")

        # Remove timezone info and convert index to date
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Validate the series
        is_valid, errors = DataValidator.validate_price_series(df, symbol)
        if not is_valid:
            logger.error(f"Validation failed for {symbol}: {errors}")
            # For now, log errors but continue (could be strict and raise)
            for error in errors:
                audit_log("VALIDATION_WARNING", symbol=symbol, error=error)

        # Convert to StockPrice objects
        prices = []
        for idx, row in df.iterrows():
            try:
                price = StockPrice(
                    symbol=symbol,
                    date=idx.date() if hasattr(idx, 'date') else idx,
                    open=round(float(row['open']), 2),
                    high=round(float(row['high']), 2),
                    low=round(float(row['low']), 2),
                    close=round(float(row['close']), 2),
                    volume=int(row['volume']),
                    adj_close=round(float(row.get('adj_close', row['close'])), 2)
                )

                # Validate individual record
                is_valid, errors = DataValidator.validate_ohlc(price)
                if is_valid:
                    prices.append(price)
                else:
                    logger.warning(f"Skipping invalid record {symbol}@{idx}: {errors}")

            except Exception as e:
                logger.warning(f"Error processing {symbol}@{idx}: {e}")

        return prices

    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        period: str = "1y"
    ) -> dict[str, list[StockPrice]]:
        """
        Fetch prices for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            period: Fallback period

        Returns:
            Dict mapping symbol to list of StockPrice
        """
        results = {}
        failed = []

        for symbol in symbols:
            try:
                prices = self.fetch_prices(symbol, start_date, end_date, period)
                if prices:
                    results[symbol] = prices
                    logger.info(f"Fetched {len(prices)} prices for {symbol}")
                else:
                    failed.append(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                failed.append(symbol)

        if failed:
            logger.warning(f"Failed to fetch {len(failed)} symbols: {failed[:10]}...")

        return results

    def fetch_intraday(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Fetch intraday data for a symbol.

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo)
            interval: Candle interval (1m, 5m, 15m, 1h)

        Returns:
            DataFrame with intraday OHLCV
        """
        yahoo_symbol = self._to_yahoo_symbol(symbol)

        try:
            # Rate limit
            self._quota.wait_and_record("yfinance")

            _p, _i = period, interval
            def _fetch_intraday():
                ticker = yf.Ticker(yahoo_symbol)
                return ticker.history(period=_p, interval=_i)
            df = yfinance_with_timeout(_fetch_intraday, timeout_seconds=30)

            if df.empty:
                logger.warning(f"No intraday data for {yahoo_symbol}")
                return pd.DataFrame()

            # Rename columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            return df

        except Exception as e:
            logger.error(f"Failed to fetch intraday {yahoo_symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current/latest price for a symbol."""
        yahoo_symbol = self._to_yahoo_symbol(symbol)

        # Check cache (short TTL for current price)
        cache_key = f"ltp:{symbol}"
        cached = self._cache.get("yfinance", cache_key)
        if cached is not None:
            return cached

        try:
            # Rate limit
            self._quota.wait_and_record("yfinance")

            def _fetch_current():
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.fast_info
                for attr in ['last_price', 'regular_market_price', 'previous_close']:
                    if hasattr(info, attr):
                        price = getattr(info, attr)
                        if price and price > 0:
                            return round(float(price), 2)
                df = ticker.history(period="1d")
                if not df.empty:
                    return round(float(df['Close'].iloc[-1]), 2)
                return None

            price = yfinance_with_timeout(_fetch_current, timeout_seconds=15)
            if price:
                self._quota.record_success("yfinance")
                self._cache.set("yfinance", cache_key, price, ttl_seconds=30)
            return price

        except Exception as e:
            self._quota.record_failure("yfinance")
            logger.error(f"Failed to get current price for {yahoo_symbol}: {e}")
            return None

    def get_stock_info(self, symbol: str) -> dict:
        """Get company info and metadata."""
        yahoo_symbol = self._to_yahoo_symbol(symbol)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'eps': info.get('trailingEps'),
                'dividend_yield': info.get('dividendYield'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            }

        except Exception as e:
            logger.error(f"Failed to get info for {yahoo_symbol}: {e}")
            return {'symbol': symbol, 'name': symbol, 'sector': 'Unknown'}
