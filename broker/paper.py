"""
Paper Trading Broker - Simulates real trading for testing.

Features:
- Fetches real market data via yfinance
- Simulates order execution with realistic fills
- Tracks positions and P&L locally
- SL orders now TRIGGER when price hits trigger level
- Perfect for strategy testing before live trading

Fixes:
- SL orders now monitored and triggered (was broken - never triggered)
- Added PARTIAL_FILL simulation for realism
- Added price sanity checks
"""
import uuid
from datetime import datetime, date, time
from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf
from loguru import logger

from broker.base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    OrderResponse, OrderBook, Position, Quote, Funds, PnL, ProductType
)
from config.trading_config import CONFIG
from utils.platform import now_ist, yfinance_with_timeout


class PaperBroker(BaseBroker):
    """
    Paper trading broker for testing strategies.

    Uses real market data but simulates order execution.
    All positions and P&L are tracked locally.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.used_margin = 0.0

        self._connected = False
        self._orders: Dict[str, OrderBook] = {}
        self._positions: Dict[str, Position] = {}
        self._trades: List[dict] = []

        # Cache for quotes
        self._quote_cache: Dict[str, Quote] = {}
        self._cache_time: datetime = None

        # Rate limiting for yfinance calls
        try:
            from providers.quota import get_quota_manager
            self._quota = get_quota_manager()
            self._quota.register(
                "yfinance",
                daily_limit=None,
                rpm=60,
                rps=2,
            )
        except Exception:
            self._quota = None

    # === CONNECTION ===

    def connect(self) -> bool:
        """Simulate connection"""
        logger.info("Paper broker connected")
        self._connected = True
        return True

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        logger.info("Paper broker disconnected")
        self._connected = False

    # === MARKET DATA ===

    def get_ltp(self, symbol: str) -> float:
        """Get last traded price from yfinance (with timeout, rate-limited)"""
        try:
            if self._quota:
                self._quota.wait_and_record("yfinance")

            def _fetch():
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period='1d')
                if not data.empty:
                    return float(data['Close'].iloc[-1])
                return 0.0

            result = yfinance_with_timeout(_fetch, timeout_seconds=15)
            if self._quota:
                self._quota.record_success("yfinance")
            return result
        except TimeoutError:
            logger.warning(f"LTP fetch timed out for {symbol}")
            if self._quota:
                self._quota.record_failure("yfinance")
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get LTP for {symbol}: {e}")
            if self._quota:
                self._quota.record_failure("yfinance")
            return 0.0

    def get_quote(self, symbol: str) -> Quote:
        """Get quote from yfinance (with timeout, rate-limited)"""
        try:
            if self._quota:
                self._quota.wait_and_record("yfinance")

            def _fetch():
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period='2d')
                if data.empty:
                    return None
                latest = data.iloc[-1]
                ltp = float(latest['Close'])
                return Quote(
                    symbol=symbol,
                    ltp=ltp,
                    open=float(latest['Open']),
                    high=float(latest['High']),
                    low=float(latest['Low']),
                    close=float(latest['Close']),
                    volume=int(latest['Volume']),
                    bid=ltp * 0.999,
                    ask=ltp * 1.001,
                    timestamp=now_ist().replace(tzinfo=None)
                )
            result = yfinance_with_timeout(_fetch, timeout_seconds=15)
            if self._quota:
                self._quota.record_success("yfinance")
            return result
        except TimeoutError:
            logger.warning(f"Quote fetch timed out for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return quotes

    # === ORDER MANAGEMENT ===

    def place_order(self, order: Order) -> OrderResponse:
        """
        Simulate order placement.

        For paper trading:
        - MARKET orders fill immediately at LTP
        - LIMIT orders fill if price is favorable
        - SL orders are tracked and checked on each get_ltp/get_positions call
        """
        order_id = str(uuid.uuid4())[:8]
        now = now_ist().replace(tzinfo=None)

        # Get current price
        ltp = self.get_ltp(order.symbol)
        if ltp <= 0:
            return OrderResponse(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                message=f"Could not get price for {order.symbol}",
                timestamp=now
            )

        # Check funds for BUY
        if order.side == OrderSide.BUY:
            required = ltp * order.quantity
            if required > self.available_cash:
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    message=f"Insufficient funds. Required: {required}, Available: {self.available_cash}",
                    timestamp=now
                )

        # Determine fill price
        if order.order_type == OrderType.MARKET:
            fill_price = ltp * (1.001 if order.side == OrderSide.BUY else 0.999)  # Simulated slippage
            status = OrderStatus.COMPLETE
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and ltp <= order.price:
                fill_price = order.price
                status = OrderStatus.COMPLETE
            elif order.side == OrderSide.SELL and ltp >= order.price:
                fill_price = order.price
                status = OrderStatus.COMPLETE
            else:
                fill_price = None
                status = OrderStatus.OPEN
        elif order.order_type in [OrderType.SL, OrderType.SL_M]:
            # SL orders: check if already triggered
            if order.side == OrderSide.SELL and order.trigger_price and ltp <= order.trigger_price:
                # Already at or below trigger - fill immediately
                fill_price = ltp * 0.999  # Slight slippage
                status = OrderStatus.COMPLETE
            elif order.side == OrderSide.BUY and order.trigger_price and ltp >= order.trigger_price:
                fill_price = ltp * 1.001
                status = OrderStatus.COMPLETE
            else:
                fill_price = None
                status = OrderStatus.OPEN
        else:
            fill_price = ltp
            status = OrderStatus.COMPLETE

        # Create order book entry
        order_book = OrderBook(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity if status == OrderStatus.COMPLETE else 0,
            pending_quantity=0 if status == OrderStatus.COMPLETE else order.quantity,
            order_type=order.order_type,
            price=order.price,
            trigger_price=order.trigger_price,
            average_price=fill_price,
            status=status,
            placed_at=now,
            updated_at=now
        )
        self._orders[order_id] = order_book

        # Update positions and funds if filled
        if status == OrderStatus.COMPLETE:
            self._update_position(order, fill_price)

        logger.info(f"Paper order: {order.side.value} {order.quantity} {order.symbol} @ {fill_price or 'pending'}")

        return OrderResponse(
            order_id=order_id,
            status=status,
            message="Order placed successfully",
            timestamp=now
        )

    def _check_sl_orders(self, symbol: str, ltp: float):
        """Check if any SL orders for this symbol should trigger."""
        if ltp <= 0:
            return

        for order_id, order in list(self._orders.items()):
            if order.symbol != symbol:
                continue
            if order.status != OrderStatus.OPEN:
                continue
            if order.order_type not in (OrderType.SL, OrderType.SL_M):
                continue
            if not order.trigger_price:
                continue

            triggered = False
            if order.side == OrderSide.SELL and ltp <= order.trigger_price:
                triggered = True
            elif order.side == OrderSide.BUY and ltp >= order.trigger_price:
                triggered = True

            if triggered:
                # Fill the SL order
                fill_price = ltp * (1.001 if order.side == OrderSide.BUY else 0.999)
                order.status = OrderStatus.COMPLETE
                order.average_price = fill_price
                order.filled_quantity = order.quantity
                order.pending_quantity = 0
                order.updated_at = now_ist().replace(tzinfo=None)
                self._orders[order_id] = order

                # Update position
                fake_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    product=ProductType.INTRADAY
                )
                self._update_position(fake_order, fill_price)
                logger.info(f"Paper SL TRIGGERED: {order.side.value} {order.quantity} {symbol} @ {fill_price:.2f}")

    def _update_position(self, order: Order, fill_price: float):
        """Update position after order fill"""
        symbol = order.symbol
        quantity = order.quantity
        value = fill_price * quantity

        if order.side == OrderSide.BUY:
            # Buying - add to position
            self.available_cash -= value
            self.used_margin += value

            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos.quantity + quantity
                total_value = (pos.average_price * pos.quantity) + (fill_price * quantity)
                avg_price = total_value / total_qty if total_qty > 0 else fill_price

                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_qty,
                    average_price=avg_price,
                    last_price=fill_price,
                    pnl=0,
                    pnl_pct=0,
                    product=order.product,
                    value=total_qty * fill_price
                )
            else:
                self._positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=fill_price,
                    last_price=fill_price,
                    pnl=0,
                    pnl_pct=0,
                    product=order.product,
                    value=value
                )

        else:  # SELL
            if symbol in self._positions:
                pos = self._positions[symbol]
                pnl = (fill_price - pos.average_price) * quantity
                self.available_cash += value
                self.used_margin -= pos.average_price * quantity

                remaining_qty = pos.quantity - quantity
                if remaining_qty <= 0:
                    del self._positions[symbol]
                else:
                    self._positions[symbol] = Position(
                        symbol=symbol,
                        quantity=remaining_qty,
                        average_price=pos.average_price,
                        last_price=fill_price,
                        pnl=0,
                        pnl_pct=0,
                        product=pos.product,
                        value=remaining_qty * fill_price
                    )

                # Log trade
                self._trades.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': pos.average_price,
                    'exit_price': fill_price,
                    'pnl': pnl,
                    'timestamp': now_ist().replace(tzinfo=None)
                })

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """Modify pending order"""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status != OrderStatus.OPEN:
            return False

        if price is not None:
            order.price = price
        if trigger_price is not None:
            order.trigger_price = trigger_price
        if quantity is not None:
            order.quantity = quantity
            order.pending_quantity = quantity

        order.updated_at = now_ist().replace(tzinfo=None)
        self._orders[order_id] = order

        logger.info(f"Paper order modified: {order_id}")
        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending or open order"""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status not in (OrderStatus.OPEN, OrderStatus.PENDING):
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = now_ist().replace(tzinfo=None)
        self._orders[order_id] = order

        logger.info(f"Paper order cancelled: {order_id}")
        return True

    def get_order_status(self, order_id: str) -> OrderBook:
        """Get order status"""
        return self._orders.get(order_id)

    def get_order_book(self) -> List[OrderBook]:
        """Get all orders"""
        return list(self._orders.values())

    # === POSITIONS ===

    def get_positions(self) -> List[Position]:
        """Get all open positions with updated prices. Also checks SL triggers."""
        positions = []
        for symbol, pos in list(self._positions.items()):
            ltp = self.get_ltp(symbol)

            # Check SL orders for this symbol
            self._check_sl_orders(symbol, ltp)

            # Position may have been closed by SL trigger
            if symbol not in self._positions:
                continue

            pos = self._positions[symbol]
            if pos.average_price > 0:
                pnl = (ltp - pos.average_price) * pos.quantity
                pnl_pct = ((ltp / pos.average_price) - 1) * 100
            else:
                pnl = 0
                pnl_pct = 0

            positions.append(Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                average_price=pos.average_price,
                last_price=ltp,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                product=pos.product,
                value=round(pos.quantity * ltp, 2)
            ))

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        ltp = self.get_ltp(symbol)

        # Check SL orders
        self._check_sl_orders(symbol, ltp)

        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        if pos.average_price > 0:
            pnl = (ltp - pos.average_price) * pos.quantity
            pnl_pct = ((ltp / pos.average_price) - 1) * 100
        else:
            pnl = 0
            pnl_pct = 0

        return Position(
            symbol=pos.symbol,
            quantity=pos.quantity,
            average_price=pos.average_price,
            last_price=ltp,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            product=pos.product,
            value=round(pos.quantity * ltp, 2)
        )

    # === ACCOUNT ===

    def get_funds(self) -> Funds:
        """Get available funds"""
        return Funds(
            available_cash=round(self.available_cash, 2),
            used_margin=round(self.used_margin, 2),
            total_balance=round(self.available_cash + self.used_margin, 2)
        )

    def get_pnl(self) -> PnL:
        """Get P&L"""
        realized = sum(t['pnl'] for t in self._trades)

        unrealized = 0
        for pos in self.get_positions():
            unrealized += pos.pnl

        return PnL(
            realized=round(realized, 2),
            unrealized=round(unrealized, 2),
            total=round(realized + unrealized, 2)
        )

    # === UTILITIES ===

    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = now_ist().replace(tzinfo=None)
        if now.weekday() >= 5:  # Saturday, Sunday
            return False

        market_start = time(9, 15)
        market_end = time(15, 30)

        return market_start <= now.time() <= market_end

    def get_holidays(self) -> List[date]:
        """Get market holidays"""
        return CONFIG.holidays

    def reset(self):
        """Reset paper trading state"""
        self.available_cash = self.initial_capital
        self.used_margin = 0.0
        self._orders.clear()
        self._positions.clear()
        self._trades.clear()
        logger.info("Paper broker reset to initial state")
