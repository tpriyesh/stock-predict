"""
Abstract Broker Interface.

All broker implementations (Upstox, Zerodha, Paper) must implement this interface.
This ensures we can easily switch brokers without changing other code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict
from enum import Enum


class OrderSide(Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderType(Enum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    SL = 'SL'           # Stop Loss Limit
    SL_M = 'SL-M'       # Stop Loss Market


class OrderStatus(Enum):
    PENDING = 'PENDING'
    OPEN = 'OPEN'
    COMPLETE = 'COMPLETE'
    PARTIAL_FILL = 'PARTIAL_FILL'
    CANCELLED = 'CANCELLED'
    REJECTED = 'REJECTED'


class ProductType(Enum):
    INTRADAY = 'I'
    DELIVERY = 'D'
    CNC = 'CNC'


@dataclass
class Order:
    """Order to be placed"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    product: ProductType = ProductType.INTRADAY
    tag: Optional[str] = None  # For internal tracking


@dataclass
class OrderResponse:
    """Response after placing order"""
    order_id: str
    status: OrderStatus
    message: str
    timestamp: datetime


@dataclass
class OrderBook:
    """Current state of an order"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    pending_quantity: int
    order_type: OrderType
    price: Optional[float]
    trigger_price: Optional[float]
    average_price: Optional[float]
    status: OrderStatus
    placed_at: datetime
    updated_at: datetime


@dataclass
class Position:
    """Open position"""
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_pct: float
    product: ProductType
    value: float


@dataclass
class Quote:
    """Market quote"""
    symbol: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime


@dataclass
class Funds:
    """Account funds"""
    available_cash: float
    used_margin: float
    total_balance: float


@dataclass
class PnL:
    """Profit & Loss"""
    realized: float
    unrealized: float
    total: float


class BaseBroker(ABC):
    """
    Abstract base class for all broker implementations.

    Every broker (Upstox, Zerodha, Paper) must implement these methods.
    """

    # === CONNECTION ===

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.
        Returns True if successful.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker"""
        pass

    # === MARKET DATA ===

    @abstractmethod
    def get_ltp(self, symbol: str) -> float:
        """
        Get Last Traded Price for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')

        Returns:
            Last traded price
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Get full quote including bid/ask.

        Args:
            symbol: Stock symbol

        Returns:
            Quote object with all details
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to Quote
        """
        pass

    # === ORDER MANAGEMENT ===

    @abstractmethod
    def place_order(self, order: Order) -> OrderResponse:
        """
        Place an order.

        Args:
            order: Order object with all details

        Returns:
            OrderResponse with order_id and status
        """
        pass

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: ID of order to modify
            price: New price (optional)
            trigger_price: New trigger price (optional)
            quantity: New quantity (optional)

        Returns:
            True if modification successful
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderBook:
        """
        Get current status of an order.

        Args:
            order_id: ID of order

        Returns:
            OrderBook with current state
        """
        pass

    @abstractmethod
    def get_order_book(self) -> List[OrderBook]:
        """
        Get all orders for today.

        Returns:
            List of all orders
        """
        pass

    # === POSITIONS ===

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position if exists, None otherwise
        """
        pass

    # === ACCOUNT ===

    @abstractmethod
    def get_funds(self) -> Funds:
        """
        Get available funds/margin.

        Returns:
            Funds object with available cash and margin used
        """
        pass

    @abstractmethod
    def get_pnl(self) -> PnL:
        """
        Get today's P&L.

        Returns:
            PnL object with realized and unrealized
        """
        pass

    # === UTILITIES ===

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        pass

    @abstractmethod
    def get_holidays(self) -> List[date]:
        """Get list of market holidays"""
        pass
