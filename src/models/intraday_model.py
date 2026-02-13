"""
Intraday prediction model - for same-day trades.
"""
from src.models.scoring import ScoringEngine, StockScore
from src.storage.models import TradeType


class IntradayModel:
    """
    Model for intraday (same-day) trading predictions.

    Focus: Quick momentum plays, mean reversion, gap fills.
    Holding period: Minutes to hours (exit by EOD).
    """

    def __init__(self):
        self.scoring_engine = ScoringEngine()
        self.trade_type = TradeType.INTRADAY

    def score(self, *args, **kwargs) -> StockScore:
        """Score a stock for intraday trading."""
        return self.scoring_engine.score_stock(
            *args,
            trade_type=self.trade_type,
            **kwargs
        )
