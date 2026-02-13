"""
Unified Stock Prediction API

Single entry point for ALL prediction use cases:
- Intraday trading
- Swing trading
- Long-term investing
- Multi-bagger hunting
- Sector analysis
- Stock screening

Each use case uses APPROPRIATE logic:
- Intraday: Technical > Fundamental
- Long-term: Fundamental > Technical
- Multi-bagger: Growth + Value + Small Cap
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger

# Import all prediction engines
from src.strategy.strategy_manager import (
    StrategyManager, StrategyType, RiskProfile,
    StrategyRecommendation, SectorAnalysis
)
from src.strategy.long_term_predictor import (
    LongTermPredictor, LongTermPrediction, InvestmentHorizon
)
from src.insights.production_predictor import ProductionPredictor


class UseCase(Enum):
    """Investment use cases."""
    INTRADAY = "intraday"            # Same-day trades
    SWING = "swing"                   # 2-10 days
    POSITIONAL = "positional"         # 2-8 weeks
    LONG_TERM_1Y = "long_term_1y"     # 1 year horizon
    LONG_TERM_3Y = "long_term_3y"     # 3 year horizon
    LONG_TERM_5Y = "long_term_5y"     # 5 year horizon
    MULTI_BAGGER = "multi_bagger"     # 5-10x potential
    SECTOR_BEST = "sector_best"       # Best in sector
    PENNY_STOCK = "penny_stock"       # Low price high potential
    QUICK_SCAN = "quick_scan"         # Fast market scan


@dataclass
class UnifiedPrediction:
    """Unified prediction output for any use case."""
    # Basic info
    symbol: str
    name: str
    sector: str
    current_price: float
    timestamp: str

    # Use case
    use_case: str
    strategy_used: str

    # Core prediction
    signal: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    confidence: float
    probability: float

    # Scores (0-100)
    overall_score: float
    technical_score: float
    fundamental_score: float
    growth_score: float
    value_score: float
    quality_score: float

    # Trade setup
    entry_price: float
    stop_loss: float
    targets: List[float]
    holding_period: str
    risk_reward: float

    # Expected outcomes
    expected_return_pct: float
    win_probability: float

    # For long-term
    target_1y: Optional[float] = None
    target_3y: Optional[float] = None
    target_5y: Optional[float] = None
    cagr_1y: Optional[float] = None
    cagr_3y: Optional[float] = None
    cagr_5y: Optional[float] = None

    # For multi-bagger
    multi_bagger_score: Optional[float] = None
    multi_bagger_factors: List[str] = field(default_factory=list)

    # Risk
    risk_level: str = "MEDIUM"
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Key metrics
    pe_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_equity: Optional[float] = None
    revenue_growth: Optional[float] = None

    # Analysis
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    recommendation: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Get text summary."""
        return f"""
{self.name} ({self.symbol}) - {self.sector}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use Case: {self.use_case}
Price: ₹{self.current_price}
Signal: {self.signal} (Confidence: {self.confidence:.0%})

📊 Scores:
  Overall: {self.overall_score:.0f}/100
  Technical: {self.technical_score:.0f} | Fundamental: {self.fundamental_score:.0f}
  Growth: {self.growth_score:.0f} | Quality: {self.quality_score:.0f}

📈 Trade Setup:
  Entry: ₹{self.entry_price}
  Stop Loss: ₹{self.stop_loss}
  Targets: {', '.join([f'₹{t}' for t in self.targets])}
  R:R = {self.risk_reward:.1f}x
  Hold: {self.holding_period}

💰 Expected:
  Return: {self.expected_return_pct:+.1f}%
  Win Rate: {self.win_probability:.0%}

{f"🚀 Multi-Bagger Score: {self.multi_bagger_score:.0f}/100" if self.multi_bagger_score else ""}
{f"📅 5Y Target: ₹{self.target_5y} (CAGR: {self.cagr_5y:+.0f}%)" if self.target_5y else ""}

✅ Bullish: {', '.join(self.bullish_factors[:3]) if self.bullish_factors else 'None'}
⚠️ Risks: {', '.join(self.bearish_factors[:3]) if self.bearish_factors else 'None'}

📝 {self.recommendation}
"""


class StockPredictor:
    """
    Unified Stock Prediction API.

    Single entry point for all prediction use cases.
    Automatically selects the right prediction engine and parameters.

    Usage:
        predictor = StockPredictor()

        # Intraday trade
        pred = predictor.predict("RELIANCE", UseCase.INTRADAY)

        # Long-term investment
        pred = predictor.predict("RELIANCE", UseCase.LONG_TERM_5Y)

        # Find multi-baggers
        picks = predictor.find_multibaggers(["IRCTC", "TATAPOWER", ...])

        # Best in sector
        picks = predictor.best_in_sector("IT")

        # Market scan
        opportunities = predictor.scan_market(UseCase.SWING)
    """

    # Stock universe for scanning
    NIFTY_50 = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
        'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'HCLTECH',
        'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN',
        'BAJFINANCE', 'WIPRO', 'ULTRACEMCO', 'NTPC', 'NESTLEIND',
        'POWERGRID', 'JSWSTEEL', 'TATASTEEL', 'ONGC', 'COALINDIA',
        'TECHM', 'HINDALCO', 'CIPLA', 'DRREDDY', 'BPCL', 'ADANIPORTS',
        'BAJAJ-AUTO', 'BAJAJFINSV', 'BRITANNIA', 'EICHERMOT', 'GRASIM',
        'HEROMOTOCO', 'INDUSINDBK', 'DIVISLAB', 'APOLLOHOSP', 'TATACONSUM',
        'SBILIFE', 'HDFCLIFE', 'TATAMOTORS', 'M&M', 'ADANIENT'
    ]

    PENNY_MIDCAP = [
        'IRCTC', 'TATAPOWER', 'NMDC', 'VEDL', 'PNB', 'CANBK', 'BANKBARODA',
        'COALINDIA', 'NHPC', 'SJVN', 'BEL', 'HAL', 'SAIL', 'NATIONALUM',
        'HINDPETRO', 'MRPL', 'GMRINFRA', 'SUZLON', 'RPOWER', 'JPPOWER',
        'PFC', 'REC', 'IRFC', 'RVNL', 'HUDCO', 'CONCOR', 'BHEL'
    ]

    def __init__(self, capital: float = 100000,
                 risk_profile: RiskProfile = RiskProfile.MODERATE):
        """
        Initialize unified predictor.

        Args:
            capital: Trading capital in INR
            risk_profile: Risk tolerance
        """
        self.capital = capital
        self.risk_profile = risk_profile

        # Initialize prediction engines
        self.strategy_manager = StrategyManager()
        self.long_term_predictor = LongTermPredictor()
        self.production_predictor = ProductionPredictor(
            capital=capital,
            enable_monitoring=True,
            enable_llm_sentiment=False  # Disable by default for speed
        )

    def predict(self, symbol: str,
                use_case: UseCase = UseCase.SWING,
                with_news: Optional[List[str]] = None) -> Optional[UnifiedPrediction]:
        """
        Generate prediction for a stock based on use case.

        Args:
            symbol: Stock symbol (e.g., RELIANCE)
            use_case: Investment use case
            with_news: Optional news headlines for sentiment

        Returns:
            UnifiedPrediction with complete analysis
        """
        try:
            logger.info(f"Predicting {symbol} for {use_case.value}...")

            # Route to appropriate predictor
            if use_case in [UseCase.INTRADAY, UseCase.SWING, UseCase.POSITIONAL]:
                return self._predict_short_term(symbol, use_case)

            elif use_case in [UseCase.LONG_TERM_1Y, UseCase.LONG_TERM_3Y,
                              UseCase.LONG_TERM_5Y]:
                return self._predict_long_term(symbol, use_case)

            elif use_case == UseCase.MULTI_BAGGER:
                return self._predict_multibagger(symbol)

            elif use_case == UseCase.PENNY_STOCK:
                return self._predict_penny_stock(symbol)

            else:
                # Default to swing trading
                return self._predict_short_term(symbol, UseCase.SWING)

        except Exception as e:
            logger.error(f"Error predicting {symbol}: {e}")
            return None

    def _predict_short_term(self, symbol: str,
                            use_case: UseCase) -> Optional[UnifiedPrediction]:
        """Predict for short-term trading."""
        # Map use case to strategy type
        strategy_map = {
            UseCase.INTRADAY: StrategyType.INTRADAY,
            UseCase.SWING: StrategyType.SWING,
            UseCase.POSITIONAL: StrategyType.POSITIONAL,
        }

        strategy = strategy_map.get(use_case, StrategyType.SWING)

        rec = self.strategy_manager.analyze(symbol, strategy, self.risk_profile)
        if not rec:
            return None

        return UnifiedPrediction(
            symbol=rec.symbol,
            name=rec.name,
            sector=rec.sector,
            current_price=rec.price,
            timestamp=datetime.now().isoformat(),
            use_case=use_case.value,
            strategy_used=strategy.value,
            signal=rec.signal,
            confidence=rec.confidence,
            probability=rec.probability_of_profit,
            overall_score=(rec.technical_score + rec.fundamental_score +
                          rec.momentum_score) / 3 * 100,
            technical_score=rec.technical_score * 100,
            fundamental_score=rec.fundamental_score * 100,
            growth_score=rec.growth_score * 100,
            value_score=rec.value_score * 100,
            quality_score=rec.quality_score * 100,
            entry_price=rec.entry_price,
            stop_loss=rec.stop_loss,
            targets=rec.targets,
            holding_period=rec.holding_period,
            risk_reward=rec.risk_reward_ratio,
            expected_return_pct=rec.expected_return_pct,
            win_probability=rec.probability_of_profit,
            risk_level=rec.risk_level,
            volatility=rec.volatility,
            bullish_factors=rec.bullish_factors,
            bearish_factors=rec.bearish_factors,
            recommendation=rec.recommendation_reason
        )

    def _predict_long_term(self, symbol: str,
                           use_case: UseCase) -> Optional[UnifiedPrediction]:
        """Predict for long-term investing."""
        # Map to horizon
        horizon_map = {
            UseCase.LONG_TERM_1Y: InvestmentHorizon.ONE_YEAR,
            UseCase.LONG_TERM_3Y: InvestmentHorizon.THREE_YEARS,
            UseCase.LONG_TERM_5Y: InvestmentHorizon.FIVE_YEARS,
        }

        horizon = horizon_map.get(use_case, InvestmentHorizon.THREE_YEARS)

        pred = self.long_term_predictor.predict(symbol, horizon)
        if not pred:
            return None

        # Map rating to signal
        signal_map = {
            'STRONG_BUY': 'STRONG_BUY',
            'BUY': 'BUY',
            'HOLD': 'HOLD',
            'SELL': 'SELL',
            'STRONG_SELL': 'STRONG_SELL'
        }

        return UnifiedPrediction(
            symbol=pred.symbol,
            name=pred.name,
            sector=pred.sector,
            current_price=pred.current_price,
            timestamp=datetime.now().isoformat(),
            use_case=use_case.value,
            strategy_used="fundamental_analysis",
            signal=signal_map.get(pred.rating, 'HOLD'),
            confidence=pred.confidence,
            probability=pred.confidence,
            overall_score=pred.overall_score,
            technical_score=pred.momentum_score,
            fundamental_score=(pred.quality_score + pred.safety_score) / 2,
            growth_score=pred.growth_score,
            value_score=pred.value_score,
            quality_score=pred.quality_score,
            entry_price=pred.current_price,
            stop_loss=pred.current_price * (1 - pred.downside_risk / 100),
            targets=[pred.target_1y, pred.target_3y, pred.target_5y],
            holding_period=f"{horizon.value} years",
            risk_reward=pred.risk_reward_ratio,
            expected_return_pct=pred.expected_cagr_3y,
            win_probability=0.5 + pred.confidence * 0.3,
            target_1y=pred.target_1y,
            target_3y=pred.target_3y,
            target_5y=pred.target_5y,
            cagr_1y=pred.expected_cagr_1y,
            cagr_3y=pred.expected_cagr_3y,
            cagr_5y=pred.expected_cagr_5y,
            multi_bagger_score=pred.multi_bagger_score,
            multi_bagger_factors=pred.multi_bagger_factors,
            risk_level="MEDIUM" if pred.safety_score > 50 else "HIGH",
            pe_ratio=pred.pe_ratio,
            roe=pred.roe,
            debt_equity=pred.debt_equity,
            revenue_growth=pred.revenue_growth,
            bullish_factors=pred.key_strengths,
            bearish_factors=pred.key_risks,
            recommendation=pred.investment_thesis
        )

    def _predict_multibagger(self, symbol: str) -> Optional[UnifiedPrediction]:
        """Predict multi-bagger potential."""
        pred = self.long_term_predictor.predict(symbol, InvestmentHorizon.FIVE_YEARS)
        if not pred:
            return None

        # Enhanced multi-bagger analysis
        mb_pred = self._predict_long_term(symbol, UseCase.LONG_TERM_5Y)
        if mb_pred:
            mb_pred.use_case = UseCase.MULTI_BAGGER.value
            mb_pred.strategy_used = "multi_bagger_hunt"

            # Boost signal if high MB score
            if pred.multi_bagger_score >= 70:
                mb_pred.signal = "STRONG_BUY"
            elif pred.multi_bagger_score >= 50:
                mb_pred.signal = "BUY"

        return mb_pred

    def _predict_penny_stock(self, symbol: str) -> Optional[UnifiedPrediction]:
        """Predict for penny/low-price stocks."""
        # Use multi-bagger logic for penny stocks
        pred = self._predict_multibagger(symbol)
        if pred:
            pred.use_case = UseCase.PENNY_STOCK.value

            # Add penny stock specific warnings
            if pred.current_price < 50:
                pred.warnings.append("Low price - higher volatility expected")
            if pred.debt_equity and pred.debt_equity > 100:
                pred.warnings.append("High debt - higher risk")

        return pred

    def find_multibaggers(self, symbols: Optional[List[str]] = None,
                          min_score: float = 50) -> List[UnifiedPrediction]:
        """
        Find potential multi-bagger stocks.

        Args:
            symbols: List of symbols to scan (default: penny/midcap list)
            min_score: Minimum multi-bagger score (0-100)

        Returns:
            List of high-potential stocks sorted by score
        """
        if symbols is None:
            symbols = self.PENNY_MIDCAP

        logger.info(f"Scanning {len(symbols)} stocks for multi-baggers...")

        results = []
        for sym in symbols:
            pred = self.predict(sym, UseCase.MULTI_BAGGER)
            if pred and pred.multi_bagger_score and pred.multi_bagger_score >= min_score:
                results.append(pred)

        # Sort by multi-bagger score
        results.sort(key=lambda x: x.multi_bagger_score or 0, reverse=True)

        return results

    def best_in_sector(self, sector: str,
                       use_case: UseCase = UseCase.LONG_TERM_3Y,
                       top_n: int = 5) -> List[UnifiedPrediction]:
        """
        Find best stocks in a sector.

        Args:
            sector: Sector name (e.g., 'IT', 'Banking')
            use_case: Investment use case
            top_n: Number of top picks

        Returns:
            Top N stocks in the sector
        """
        sector_analysis = self.strategy_manager.analyze_sector(sector)

        results = []
        for pick in sector_analysis.top_picks[:top_n]:
            pred = self.predict(pick.symbol, use_case)
            if pred:
                results.append(pred)

        return results

    def scan_market(self, use_case: UseCase = UseCase.SWING,
                    symbols: Optional[List[str]] = None,
                    min_confidence: float = 0.6) -> List[UnifiedPrediction]:
        """
        Scan market for opportunities.

        Args:
            use_case: Investment use case
            symbols: Symbols to scan (default: NIFTY 50)
            min_confidence: Minimum confidence threshold

        Returns:
            List of opportunities sorted by confidence
        """
        if symbols is None:
            symbols = self.NIFTY_50

        logger.info(f"Scanning {len(symbols)} stocks for {use_case.value}...")

        results = []
        for sym in symbols:
            pred = self.predict(sym, use_case)
            if pred and pred.confidence >= min_confidence:
                if pred.signal in ['BUY', 'STRONG_BUY']:
                    results.append(pred)

        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)

        return results

    def compare_stocks(self, symbols: List[str],
                       use_case: UseCase = UseCase.LONG_TERM_3Y
                       ) -> pd.DataFrame:
        """
        Compare multiple stocks side by side.

        Args:
            symbols: List of symbols to compare
            use_case: Investment use case

        Returns:
            DataFrame with comparison
        """
        data = []
        for sym in symbols:
            pred = self.predict(sym, use_case)
            if pred:
                data.append({
                    'Symbol': pred.symbol,
                    'Price': pred.current_price,
                    'Signal': pred.signal,
                    'Confidence': f"{pred.confidence:.0%}",
                    'Overall': f"{pred.overall_score:.0f}",
                    'Technical': f"{pred.technical_score:.0f}",
                    'Fundamental': f"{pred.fundamental_score:.0f}",
                    'Growth': f"{pred.growth_score:.0f}",
                    'Value': f"{pred.value_score:.0f}",
                    'Quality': f"{pred.quality_score:.0f}",
                    'R:R': f"{pred.risk_reward:.1f}x",
                    'MB Score': f"{pred.multi_bagger_score or 0:.0f}" if pred.multi_bagger_score else "-"
                })

        df = pd.DataFrame(data)
        return df.sort_values('Overall', ascending=False)

    def get_portfolio_allocation(self, symbols: List[str],
                                 use_case: UseCase = UseCase.LONG_TERM_3Y
                                 ) -> Dict[str, Any]:
        """
        Get recommended portfolio allocation.

        Args:
            symbols: Symbols to include
            use_case: Investment strategy

        Returns:
            Portfolio allocation with weights
        """
        predictions = []
        for sym in symbols:
            pred = self.predict(sym, use_case)
            if pred and pred.signal in ['BUY', 'STRONG_BUY']:
                predictions.append(pred)

        if not predictions:
            return {'error': 'No buy recommendations found'}

        # Calculate weights based on confidence and quality
        total_score = sum(p.overall_score * p.confidence for p in predictions)

        allocations = []
        for pred in predictions:
            weight = (pred.overall_score * pred.confidence) / total_score
            amount = self.capital * weight
            qty = int(amount / pred.current_price)

            allocations.append({
                'symbol': pred.symbol,
                'weight': round(weight * 100, 1),
                'amount': round(amount, 2),
                'quantity': qty,
                'entry': pred.entry_price,
                'stop_loss': pred.stop_loss,
                'target': pred.targets[0] if pred.targets else None,
                'signal': pred.signal,
                'confidence': pred.confidence
            })

        return {
            'capital': self.capital,
            'use_case': use_case.value,
            'allocations': allocations,
            'total_stocks': len(allocations),
            'generated_at': datetime.now().isoformat()
        }


def demo():
    """Demonstrate unified API."""
    print("=" * 70)
    print("UNIFIED STOCK PREDICTION API DEMO")
    print("=" * 70)

    predictor = StockPredictor(capital=100000)

    # Test different use cases
    symbol = "RELIANCE"

    print(f"\n--- Comparing {symbol} across use cases ---\n")

    use_cases = [
        UseCase.INTRADAY,
        UseCase.SWING,
        UseCase.LONG_TERM_3Y,
        UseCase.MULTI_BAGGER
    ]

    for uc in use_cases:
        pred = predictor.predict(symbol, uc)
        if pred:
            print(f"{uc.value.upper()}: {pred.signal} "
                  f"(Conf: {pred.confidence:.0%}, Score: {pred.overall_score:.0f})")
            if uc == UseCase.LONG_TERM_3Y:
                print(f"  3Y Target: ₹{pred.target_3y} (CAGR: {pred.cagr_3y:+.0f}%)")
            elif uc == UseCase.MULTI_BAGGER:
                print(f"  MB Score: {pred.multi_bagger_score:.0f}/100")

    # Find multi-baggers
    print("\n--- Top Multi-Baggers ---\n")
    multibaggers = predictor.find_multibaggers(min_score=40)
    for mb in multibaggers[:3]:
        print(f"  {mb.symbol}: Score {mb.multi_bagger_score:.0f}, "
              f"5Y Target ₹{mb.target_5y} ({mb.cagr_5y:+.0f}% CAGR)")

    # Compare IT sector stocks
    print("\n--- IT Sector Comparison ---\n")
    it_stocks = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']
    comparison = predictor.compare_stocks(it_stocks, UseCase.LONG_TERM_3Y)
    print(comparison.to_string(index=False))

    # Portfolio allocation
    print("\n--- Portfolio Allocation (₹1,00,000) ---\n")
    allocation = predictor.get_portfolio_allocation(
        ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK'],
        UseCase.LONG_TERM_3Y
    )
    for alloc in allocation.get('allocations', []):
        print(f"  {alloc['symbol']}: {alloc['weight']}% "
              f"(₹{alloc['amount']:,.0f}, {alloc['quantity']} shares)")


if __name__ == "__main__":
    demo()
