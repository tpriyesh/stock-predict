"""
Survivorship Bias Handler

CRITICAL: Backtests using only currently-listed stocks are BIASED.

Problem:
- Stocks that crashed, delisted, or were acquired are excluded
- This inflates historical returns by 2-4% annually
- Selection of "winners" creates false confidence

Solution:
1. Track historical index constituents
2. Include delisted stocks with proper returns (often -100%)
3. Adjust for corporate actions (splits, mergers)
4. Flag periods where data may be biased

This module provides:
- Historical NSE/BSE index constituent tracking
- Delisted stock database
- Survivorship-adjusted returns calculation
- Bias quantification metrics
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import json
import os


class DelistReason(Enum):
    """Reasons for stock delisting."""
    BANKRUPTCY = "bankruptcy"           # Company went bankrupt
    MERGER = "merger"                   # Acquired/merged
    VOLUNTARY = "voluntary"             # Voluntary delisting
    REGULATORY = "regulatory"           # Regulatory action
    LOW_LIQUIDITY = "low_liquidity"     # Insufficient trading
    NON_COMPLIANCE = "non_compliance"   # Listing requirements not met
    UNKNOWN = "unknown"


@dataclass
class DelistedStock:
    """Information about a delisted stock."""
    symbol: str
    name: str
    delist_date: date
    reason: DelistReason
    last_price: float
    peak_price: float           # Highest price before delisting
    return_from_peak: float     # Return from peak to delist (usually negative)
    index_member: List[str]     # Which indices it was part of


@dataclass
class SurvivorshipAdjustment:
    """Adjustments to apply for survivorship bias."""
    period_start: date
    period_end: date
    stocks_in_universe: int
    stocks_delisted: int
    average_delist_return: float
    bias_adjustment_factor: float  # Multiply returns by this
    delisted_stocks: List[DelistedStock]


class SurvivorshipBiasHandler:
    """
    Handles survivorship bias in backtesting.

    Uses historical index constituent data and tracks delistings.
    """

    # Known NSE delistings with significant impact (sample data)
    # In production, this would be fetched from a comprehensive database
    KNOWN_DELISTINGS: List[Dict] = [
        # Major delistings from NIFTY/indices
        {"symbol": "RCOM", "name": "Reliance Communications", "delist_date": "2022-12-20",
         "reason": "bankruptcy", "last_price": 1.75, "peak_price": 844.0, "indices": ["NIFTY50"]},
        {"symbol": "JPASSOCIAT", "name": "JP Associates", "delist_date": "2023-08-01",
         "reason": "regulatory", "last_price": 5.0, "peak_price": 339.0, "indices": ["NIFTY500"]},
        {"symbol": "UNITECH", "name": "Unitech Ltd", "delist_date": "2020-01-01",
         "reason": "regulatory", "last_price": 1.0, "peak_price": 545.0, "indices": ["NIFTY50"]},
        {"symbol": "SUZLON", "name": "Suzlon Energy", "delist_date": "2020-06-01",
         "reason": "regulatory", "last_price": 3.0, "peak_price": 440.0, "indices": ["NIFTY500"]},
        {"symbol": "RPOWER", "name": "Reliance Power", "delist_date": "2023-01-01",
         "reason": "regulatory", "last_price": 12.0, "peak_price": 430.0, "indices": ["NIFTY100"]},
        {"symbol": "GTLINFRA", "name": "GTL Infrastructure", "delist_date": "2021-01-01",
         "reason": "bankruptcy", "last_price": 0.5, "peak_price": 55.0, "indices": ["NIFTY500"]},
        {"symbol": "KSOILS", "name": "K S Oils", "delist_date": "2019-01-01",
         "reason": "fraud", "last_price": 0.0, "peak_price": 72.0, "indices": ["NIFTY500"]},
        {"symbol": "VIDEOIND", "name": "Videocon Industries", "delist_date": "2021-06-01",
         "reason": "bankruptcy", "last_price": 2.0, "peak_price": 840.0, "indices": ["NIFTY200"]},
        {"symbol": "LEEL", "name": "LEEL Electricals", "delist_date": "2022-01-01",
         "reason": "merger", "last_price": 150.0, "peak_price": 320.0, "indices": ["NIFTY500"]},
        {"symbol": "CASTEXTECH", "name": "Castex Technologies", "delist_date": "2020-01-01",
         "reason": "bankruptcy", "last_price": 1.0, "peak_price": 155.0, "indices": ["NIFTY500"]},
    ]

    # Historical NIFTY 50 changes (sample - would need complete history)
    NIFTY50_CHANGES: List[Dict] = [
        {"date": "2024-09-30", "added": ["TRENT"], "removed": ["WIPRO"]},
        {"date": "2024-03-28", "added": ["SHRIRAMFIN"], "removed": ["UPL"]},
        {"date": "2023-09-29", "added": ["LTIM", "JIOFIN"], "removed": ["APOLLOHOSP", "ADANIPORTS"]},
        {"date": "2023-03-31", "added": ["LTIM"], "removed": ["HDFC"]},
        {"date": "2022-09-30", "added": ["ADANIENT"], "removed": ["SHREECEM"]},
        {"date": "2022-03-31", "added": ["APOLLOHOSP"], "removed": ["IOC"]},
        {"date": "2021-09-24", "added": ["TATACONSUM"], "removed": ["GAIL"]},
        {"date": "2020-09-25", "added": ["SBI", "TATAMOTORS"], "removed": ["VEDL", "ZEEL"]},
    ]

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or '/tmp/survivorship_data'
        os.makedirs(self.data_dir, exist_ok=True)
        self.delisted_stocks: List[DelistedStock] = []
        self._load_delistings()

    def _load_delistings(self):
        """Load delisting data."""
        for d in self.KNOWN_DELISTINGS:
            delist_date = datetime.strptime(d['delist_date'], '%Y-%m-%d').date()
            return_from_peak = (d['last_price'] - d['peak_price']) / d['peak_price']

            self.delisted_stocks.append(DelistedStock(
                symbol=d['symbol'],
                name=d['name'],
                delist_date=delist_date,
                reason=DelistReason(d['reason']),
                last_price=d['last_price'],
                peak_price=d['peak_price'],
                return_from_peak=return_from_peak,
                index_member=d['indices']
            ))

    def get_historical_universe(self,
                                 as_of_date: date,
                                 index: str = "NIFTY500") -> Set[str]:
        """
        Get stock universe as it existed on a historical date.

        Args:
            as_of_date: The historical date
            index: Index to get constituents for

        Returns:
            Set of symbols that were in the index on that date
        """
        # Start with current universe (would need actual historical data)
        # This is a simplified implementation
        current_symbols = self._get_current_universe(index)

        # Add back stocks that were removed after as_of_date
        for change in self.NIFTY50_CHANGES:
            change_date = datetime.strptime(change['date'], '%Y-%m-%d').date()
            if change_date > as_of_date:
                # This change happened AFTER our date, so reverse it
                for removed in change.get('removed', []):
                    current_symbols.add(removed)
                for added in change.get('added', []):
                    current_symbols.discard(added)

        # Add delisted stocks that were still listed on as_of_date
        for dstock in self.delisted_stocks:
            if dstock.delist_date > as_of_date and index in dstock.index_member:
                current_symbols.add(dstock.symbol)

        return current_symbols

    def _get_current_universe(self, index: str) -> Set[str]:
        """Get current index constituents."""
        # Would fetch from config/symbols.json in production
        if index == "NIFTY50":
            return {
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "LT", "HCLTECH", "AXISBANK", "ASIANPAINT", "MARUTI",
                "SUNPHARMA", "TITAN", "DMART", "ULTRACEMCO", "BAJFINANCE",
                "WIPRO", "NESTLEIND", "M&M", "NTPC", "POWERGRID",
                "TATAMOTORS", "ADANIENT", "JSWSTEEL", "TATASTEEL", "TECHM",
                "BAJAJ-AUTO", "ONGC", "INDUSINDBK", "COALINDIA", "HINDALCO",
                "GRASIM", "DRREDDY", "BPCL", "EICHERMOT", "CIPLA",
                "DIVISLAB", "SBILIFE", "HDFCLIFE", "APOLLOHOSP", "TATACONSUM",
                "BRITANNIA", "HEROMOTOCO", "UPL", "BAJAJFINSV", "ADANIPORTS"
            }
        else:
            return set()  # Would return appropriate index constituents

    def calculate_survivorship_adjustment(self,
                                           start_date: date,
                                           end_date: date,
                                           index: str = "NIFTY500") -> SurvivorshipAdjustment:
        """
        Calculate survivorship bias adjustment for a backtest period.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            index: Index being backtested

        Returns:
            SurvivorshipAdjustment with bias correction factors
        """
        # Find stocks delisted during this period
        delisted_in_period = [
            d for d in self.delisted_stocks
            if start_date <= d.delist_date <= end_date and index in d.index_member
        ]

        # Get universe size at start
        universe_at_start = self.get_historical_universe(start_date, index)

        if not delisted_in_period:
            return SurvivorshipAdjustment(
                period_start=start_date,
                period_end=end_date,
                stocks_in_universe=len(universe_at_start),
                stocks_delisted=0,
                average_delist_return=0,
                bias_adjustment_factor=1.0,
                delisted_stocks=[]
            )

        # Calculate average return of delisted stocks
        delist_returns = [d.return_from_peak for d in delisted_in_period]
        avg_delist_return = np.mean(delist_returns)

        # Calculate bias adjustment
        # If 2% of stocks delisted with -90% return, bias ≈ 0.02 * 0.90 = 1.8% upward bias
        delist_fraction = len(delisted_in_period) / max(1, len(universe_at_start))
        bias_per_year = delist_fraction * abs(avg_delist_return)

        # Annualize
        years = (end_date - start_date).days / 365.25
        total_bias = bias_per_year * years

        # Adjustment factor (to reduce inflated returns)
        adjustment_factor = 1 / (1 + total_bias)

        logger.info(f"Survivorship adjustment: {len(delisted_in_period)} delistings, "
                   f"avg return {avg_delist_return:.1%}, adjustment factor {adjustment_factor:.3f}")

        return SurvivorshipAdjustment(
            period_start=start_date,
            period_end=end_date,
            stocks_in_universe=len(universe_at_start),
            stocks_delisted=len(delisted_in_period),
            average_delist_return=avg_delist_return,
            bias_adjustment_factor=adjustment_factor,
            delisted_stocks=delisted_in_period
        )

    def adjust_backtest_returns(self,
                                 returns: List[float],
                                 start_date: date,
                                 end_date: date,
                                 index: str = "NIFTY500") -> Tuple[List[float], Dict]:
        """
        Adjust backtest returns for survivorship bias.

        Args:
            returns: List of trade returns (as percentages)
            start_date: Backtest start
            end_date: Backtest end
            index: Index universe

        Returns:
            Tuple of (adjusted_returns, adjustment_info)
        """
        adjustment = self.calculate_survivorship_adjustment(start_date, end_date, index)

        # Apply adjustment to each return
        adjusted_returns = [r * adjustment.bias_adjustment_factor for r in returns]

        # Also add "phantom trades" for delisted stocks
        # These represent what would have happened if we held delisted stocks
        for dstock in adjustment.delisted_stocks:
            # Add a negative return representing a typical delisting scenario
            phantom_return = dstock.return_from_peak * 100  # Convert to percentage
            adjusted_returns.append(phantom_return)

        info = {
            'original_trade_count': len(returns),
            'adjusted_trade_count': len(adjusted_returns),
            'phantom_trades_added': len(adjustment.delisted_stocks),
            'adjustment_factor': adjustment.bias_adjustment_factor,
            'original_avg_return': np.mean(returns) if returns else 0,
            'adjusted_avg_return': np.mean(adjusted_returns) if adjusted_returns else 0,
            'survivorship_bias_estimated': (np.mean(returns) - np.mean(adjusted_returns)) if returns and adjusted_returns else 0,
            'delisted_symbols': [d.symbol for d in adjustment.delisted_stocks]
        }

        return adjusted_returns, info

    def get_delisting_risk_score(self, symbol: str, df: pd.DataFrame) -> float:
        """
        Calculate risk score for potential delisting.

        Higher score = higher risk of delisting/distress.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame

        Returns:
            Risk score (0-1)
        """
        score = 0.0

        if df.empty or len(df) < 50:
            return 0.5  # Insufficient data

        # 1. Check if price is near all-time low
        current_price = df['close'].iloc[-1]
        all_time_high = df['close'].max()
        drawdown = (all_time_high - current_price) / all_time_high

        if drawdown > 0.9:
            score += 0.3  # >90% from ATH is very risky
        elif drawdown > 0.7:
            score += 0.2
        elif drawdown > 0.5:
            score += 0.1

        # 2. Check volume trend (declining volume is concerning)
        recent_vol = df['volume'].tail(20).mean()
        older_vol = df['volume'].tail(100).head(80).mean()

        if older_vol > 0:
            vol_ratio = recent_vol / older_vol
            if vol_ratio < 0.3:
                score += 0.2  # Volume collapsed
            elif vol_ratio < 0.5:
                score += 0.1

        # 3. Check if penny stock
        if current_price < 10:
            score += 0.15
        elif current_price < 50:
            score += 0.05

        # 4. Check for sustained downtrend
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else sma_50

        if pd.notna(sma_50) and pd.notna(sma_200):
            if current_price < sma_50 < sma_200:
                score += 0.1  # Death cross territory

        # 5. Check for extreme volatility (sign of distress)
        returns = df['close'].pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            if volatility > 1.0:  # 100% annualized volatility
                score += 0.15

        return min(1.0, score)

    def should_exclude_stock(self,
                              symbol: str,
                              df: pd.DataFrame,
                              max_risk_score: float = 0.5) -> Tuple[bool, str]:
        """
        Determine if stock should be excluded due to delisting risk.

        Args:
            symbol: Stock symbol
            df: OHLCV data
            max_risk_score: Maximum acceptable risk score

        Returns:
            Tuple of (should_exclude, reason)
        """
        risk_score = self.get_delisting_risk_score(symbol, df)

        if risk_score > max_risk_score:
            return True, f"High delisting risk score: {risk_score:.2f}"

        # Check if already in known distressed list
        for dstock in self.delisted_stocks:
            if dstock.symbol == symbol:
                return True, f"Stock was previously delisted ({dstock.reason.value})"

        return False, ""


def calculate_bias_adjusted_metrics(trades: List[Dict],
                                     start_date: date,
                                     end_date: date) -> Dict:
    """
    Calculate backtest metrics with survivorship bias adjustment.

    Args:
        trades: List of trade dictionaries with 'pnl_pct' key
        start_date: Backtest start
        end_date: Backtest end

    Returns:
        Dict with both raw and adjusted metrics
    """
    handler = SurvivorshipBiasHandler()

    returns = [t['pnl_pct'] for t in trades]
    adjusted_returns, adjustment_info = handler.adjust_backtest_returns(
        returns, start_date, end_date
    )

    raw_metrics = {
        'total_trades': len(returns),
        'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0,
        'avg_return': np.mean(returns) if returns else 0,
        'total_return': np.sum(returns) if returns else 0,
    }

    adjusted_metrics = {
        'total_trades': len(adjusted_returns),
        'win_rate': len([r for r in adjusted_returns if r > 0]) / len(adjusted_returns) if adjusted_returns else 0,
        'avg_return': np.mean(adjusted_returns) if adjusted_returns else 0,
        'total_return': np.sum(adjusted_returns) if adjusted_returns else 0,
    }

    return {
        'raw': raw_metrics,
        'adjusted': adjusted_metrics,
        'adjustment_info': adjustment_info,
        'bias_warning': adjustment_info['survivorship_bias_estimated'] > 1.0  # >1% bias is significant
    }
