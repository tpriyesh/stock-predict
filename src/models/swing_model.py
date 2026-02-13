"""
Swing trading prediction model - for multi-day holds.
"""
from src.models.scoring import ScoringEngine, StockScore
from src.storage.models import TradeType


class SwingModel:
    """
    Model for swing trading predictions.

    Focus: Trend following, breakouts, mean reversion on larger timeframes.
    Holding period: 2-10 days.
    """

    def __init__(self):
        self.scoring_engine = ScoringEngine()
        self.trade_type = TradeType.SWING

    def score(self, *args, **kwargs) -> StockScore:
        """Score a stock for swing trading."""
        return self.scoring_engine.score_stock(
            *args,
            trade_type=self.trade_type,
            **kwargs
        )
