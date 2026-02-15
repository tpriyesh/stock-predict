"""
Market-wide indicators: VIX, FII/DII flows, sector indices.

Uses ResponseCache (1-hour TTL) and quota manager to avoid
redundant yfinance calls — market indices change slowly.
"""
import os
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
from loguru import logger

from src.data.price_fetcher import PriceFetcher
from providers.cache import get_response_cache
from providers.quota import get_quota_manager
from utils.platform import yfinance_with_timeout, now_ist


# Default cache TTL for market index data (1 hour)
_INDEX_CACHE_TTL = int(os.getenv("MARKET_INDEX_CACHE_TTL", "3600"))


class MarketIndicators:
    """
    Fetches market-wide indicators for regime detection and context.
    """

    # Index symbols for Yahoo Finance
    INDICES = {
        'NIFTY50': '^NSEI',
        'NIFTYBANK': '^NSEBANK',
        'NIFTYIT': '^CNXIT',
        'NIFTYPHARMA': '^CNXPHARMA',
        'NIFTYMETAL': '^CNXMETAL',
        'NIFTYAUTO': '^CNXAUTO',
        'NIFTYFMCG': '^CNXFMCG',
        'NIFTYREALTY': '^CNXREALTY',
        'NIFTYENERGY': '^CNXENERGY',
        'NIFTYPSE': '^CNXPSE',
        'INDIAVIX': '^INDIAVIX'
    }

    # Sector mapping for stocks
    SECTOR_INDICES = {
        'IT': 'NIFTYIT',
        'Banking': 'NIFTYBANK',
        'Finance': 'NIFTYBANK',
        'Auto': 'NIFTYAUTO',
        'Pharma': 'NIFTYPHARMA',
        'Oil_Gas': 'NIFTYENERGY',
        'Metals': 'NIFTYMETAL',
        'FMCG': 'NIFTYFMCG',
        'Infra': 'NIFTYREALTY',
        'Power': 'NIFTYPSE',
    }

    def __init__(self):
        self.price_fetcher = PriceFetcher()
        self._cache = get_response_cache()
        self._quota = get_quota_manager()

    def get_index_data(
        self,
        index_name: str,
        period: str = "1mo"
    ) -> pd.DataFrame:
        """
        Get historical data for an index (cached for 1 hour).

        Args:
            index_name: Name from INDICES dict (e.g., 'NIFTY50', 'INDIAVIX')
            period: Data period

        Returns:
            DataFrame with OHLCV data
        """
        if index_name not in self.INDICES:
            logger.warning(f"Unknown index: {index_name}")
            return pd.DataFrame()

        # Check cache first
        cache_key = f"index:{index_name}:{period}"
        cached = self._cache.get("market_index", cache_key)
        if cached is not None:
            return cached

        yahoo_symbol = self.INDICES[index_name]

        try:
            # Rate limit via quota manager
            allowed, reason = self._quota.can_request("yfinance")
            if not allowed:
                logger.debug(f"yfinance blocked for {index_name}: {reason}")
                return pd.DataFrame()

            self._quota.wait_and_record("yfinance")

            def _fetch():
                ticker = yf.Ticker(yahoo_symbol)
                return ticker.history(period=period)

            df = yfinance_with_timeout(_fetch, timeout_seconds=30)

            if df.empty:
                logger.warning(f"No data for {index_name}")
                self._quota.record_success("yfinance")
                return pd.DataFrame()

            self._quota.record_success("yfinance")

            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Cache for 1 hour — market indices change slowly
            self._cache.set("market_index", cache_key, df,
                            ttl_seconds=_INDEX_CACHE_TTL)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch {index_name}: {e}")
            try:
                self._quota.record_failure("yfinance")
            except Exception:
                pass
            return pd.DataFrame()

    def get_vix(self, period: str = "1mo") -> pd.DataFrame:
        """Get India VIX data (fear gauge)."""
        return self.get_index_data('INDIAVIX', period)

    def get_current_vix(self) -> Optional[float]:
        """Get current VIX value."""
        try:
            df = self.get_vix("5d")
            if not df.empty:
                return round(float(df['close'].iloc[-1]), 2)
            return None
        except Exception as e:
            logger.error(f"Failed to get current VIX: {e}")
            return None

    def get_market_regime(self) -> dict:
        """
        Determine current market regime based on multiple indicators.

        Returns:
            Dict with regime classification and supporting data
        """
        regime = {
            'timestamp': now_ist().replace(tzinfo=None).isoformat(),
            'overall': 'NEUTRAL',
            'vix_level': None,
            'vix_regime': 'NORMAL',
            'nifty_trend': 'SIDEWAYS',
            'vix_trend': 'STABLE',
            'sector_leaders': [],
            'sector_laggards': [],
        }

        # Get VIX
        vix = self.get_current_vix()
        if vix:
            regime['vix_level'] = vix
            if vix < 13:
                regime['vix_regime'] = 'LOW_FEAR'  # Complacency
            elif vix > 20:
                regime['vix_regime'] = 'HIGH_FEAR'  # Fear
            else:
                regime['vix_regime'] = 'NORMAL'

        # VIX trend direction (compare to 5-day SMA)
        vix_df = self.get_vix("1mo")
        if not vix_df.empty and len(vix_df) >= 5:
            vix_current = vix_df['close'].iloc[-1]
            vix_sma5 = vix_df['close'].tail(5).mean()
            pct_diff = (vix_current - vix_sma5) / vix_sma5
            if pct_diff > 0.05:
                regime['vix_trend'] = 'RISING'
            elif pct_diff < -0.05:
                regime['vix_trend'] = 'FALLING'
            else:
                regime['vix_trend'] = 'STABLE'

        # Get NIFTY trend
        nifty = self.get_index_data('NIFTY50', '3mo')
        if not nifty.empty:
            current = nifty['close'].iloc[-1]
            sma20 = nifty['close'].tail(20).mean()
            sma50 = nifty['close'].tail(50).mean()

            if current > sma20 > sma50:
                regime['nifty_trend'] = 'BULLISH'
            elif current < sma20 < sma50:
                regime['nifty_trend'] = 'BEARISH'
            else:
                regime['nifty_trend'] = 'SIDEWAYS'

        # Sector rotation analysis
        sector_returns = self._get_sector_returns()
        if sector_returns:
            sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
            regime['sector_leaders'] = [s[0] for s in sorted_sectors[:3]]
            regime['sector_laggards'] = [s[0] for s in sorted_sectors[-3:]]

        # Overall regime — explicit handling of mixed signals
        if regime['vix_regime'] == 'LOW_FEAR' and regime['nifty_trend'] == 'BULLISH':
            regime['overall'] = 'RISK_ON'
        elif regime['vix_regime'] == 'HIGH_FEAR' and regime['nifty_trend'] == 'BEARISH':
            regime['overall'] = 'RISK_OFF'
        elif regime['vix_regime'] == 'HIGH_FEAR' and regime['nifty_trend'] in ('BULLISH', 'SIDEWAYS'):
            regime['overall'] = 'CAUTIOUS'  # Mixed: high fear but not bearish
        elif regime['vix_regime'] == 'LOW_FEAR' and regime['nifty_trend'] == 'BEARISH':
            regime['overall'] = 'CAUTIOUS'  # Mixed: low fear but bearish
        elif regime['nifty_trend'] == 'BEARISH':
            regime['overall'] = 'RISK_OFF'  # Bearish without VIX data
        else:
            regime['overall'] = 'NEUTRAL'

        return regime

    def _get_sector_returns(self, period: str = "1mo") -> dict[str, float]:
        """Get returns for all sector indices."""
        returns = {}

        for sector_name, index_name in self.SECTOR_INDICES.items():
            try:
                df = self.get_index_data(index_name, period)
                if not df.empty and len(df) > 1:
                    start_price = df['close'].iloc[0]
                    end_price = df['close'].iloc[-1]
                    ret = (end_price - start_price) / start_price * 100
                    returns[sector_name] = round(ret, 2)
            except Exception as e:
                logger.warning(f"Could not get returns for {sector_name}: {e}")

        return returns

    def get_sector_strength(self, sector: str) -> dict:
        """
        Get strength metrics for a specific sector.

        Args:
            sector: Sector name (e.g., 'IT', 'Banking')

        Returns:
            Dict with strength metrics
        """
        index_name = self.SECTOR_INDICES.get(sector)
        if not index_name:
            return {'sector': sector, 'strength': 'UNKNOWN', 'score': 0.5}

        try:
            df = self.get_index_data(index_name, '3mo')
            if df.empty:
                return {'sector': sector, 'strength': 'UNKNOWN', 'score': 0.5}

            current = df['close'].iloc[-1]
            sma20 = df['close'].tail(20).mean()
            sma50 = df['close'].tail(50).mean()

            # Calculate relative strength
            rs_20 = (current - sma20) / sma20
            rs_50 = (current - sma50) / sma50

            # Score from 0 to 1
            score = 0.5 + (rs_20 * 2 + rs_50) / 6
            score = max(0, min(1, score))

            if score > 0.6:
                strength = 'BULLISH'
            elif score < 0.4:
                strength = 'BEARISH'
            else:
                strength = 'NEUTRAL'

            # Get recent momentum
            week_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
            month_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100

            return {
                'sector': sector,
                'index': index_name,
                'strength': strength,
                'score': round(score, 3),
                'current_level': round(current, 2),
                'sma_20': round(sma20, 2),
                'sma_50': round(sma50, 2),
                'week_return_pct': round(week_return, 2),
                'month_return_pct': round(month_return, 2)
            }

        except Exception as e:
            logger.error(f"Failed to get sector strength for {sector}: {e}")
            return {'sector': sector, 'strength': 'UNKNOWN', 'score': 0.5}

    def get_all_sector_strengths(self) -> list[dict]:
        """Get strength metrics for all tracked sectors."""
        results = []
        for sector in self.SECTOR_INDICES.keys():
            results.append(self.get_sector_strength(sector))
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def get_nifty_levels(self) -> dict:
        """Get key NIFTY 50 levels for context."""
        try:
            df = self.get_index_data('NIFTY50', '1y')
            if df.empty:
                return {}

            current = df['close'].iloc[-1]

            return {
                'current': round(current, 2),
                'day_high': round(df['high'].iloc[-1], 2),
                'day_low': round(df['low'].iloc[-1], 2),
                'week_high': round(df['high'].tail(5).max(), 2),
                'week_low': round(df['low'].tail(5).min(), 2),
                'month_high': round(df['high'].tail(22).max(), 2),
                'month_low': round(df['low'].tail(22).min(), 2),
                'year_high': round(df['high'].max(), 2),
                'year_low': round(df['low'].min(), 2),
                'sma_20': round(df['close'].tail(20).mean(), 2),
                'sma_50': round(df['close'].tail(50).mean(), 2),
                'sma_200': round(df['close'].tail(200).mean(), 2),
            }

        except Exception as e:
            logger.error(f"Failed to get NIFTY levels: {e}")
            return {}
