"""
Broker module - handles all broker API interactions.

Supports:
- Upstox
- Zerodha (Kite Connect)
- Paper trading (for testing)

Usage:
    from broker import get_broker

    broker = get_broker('paper')  # or 'upstox' or 'zerodha'
    broker.connect()
    ltp = broker.get_ltp('RELIANCE')
"""

from broker.base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderResponse,
    OrderBook,
    Position,
    Quote,
    Funds,
    PnL,
    ProductType
)


def get_broker(broker_type: str = 'paper') -> BaseBroker:
    """
    Factory function to get broker instance.

    Args:
        broker_type: 'paper', 'upstox', or 'zerodha'

    Returns:
        Broker instance
    """
    if broker_type == 'paper':
        from broker.paper import PaperBroker
        return PaperBroker()
    elif broker_type == 'upstox':
        from broker.upstox import UpstoxBroker
        return UpstoxBroker()
    elif broker_type == 'zerodha':
        from broker.zerodha import ZerodhaBroker
        return ZerodhaBroker()
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")


__all__ = [
    'get_broker',
    'BaseBroker',
    'Order',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'OrderResponse',
    'OrderBook',
    'Position',
    'Quote',
    'Funds',
    'PnL',
    'ProductType'
]
