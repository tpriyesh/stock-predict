"""
Upstox Broker - Real trading integration with Upstox API.

Features:
- OAuth2 authentication with token refresh
- Rate-limited API calls
- Real-time market data
- Order placement and management
"""
import os
import json
import time
import requests
from datetime import datetime, date, time as dt_time, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from broker.base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    OrderResponse, OrderBook, Position, Quote, Funds, PnL, ProductType
)
from utils.rate_limiter import BrokerRateLimiter, rate_limited
from utils.error_handler import (
    BrokerException, RateLimitException, AuthException,
    retry_with_backoff, CircuitBreaker
)
from utils.platform import now_ist, today_ist


class UpstoxBroker(BaseBroker):
    """
    Upstox broker implementation.

    Requires environment variables:
    - UPSTOX_API_KEY: Your API key
    - UPSTOX_API_SECRET: Your API secret
    - UPSTOX_ACCESS_TOKEN: Access token (obtained via OAuth)
    """

    BASE_URL = "https://api.upstox.com/v2"

    # Product type mapping
    PRODUCT_MAP = {
        ProductType.INTRADAY: "I",
        ProductType.DELIVERY: "D",
        ProductType.CNC: "D",
    }

    # Order type mapping
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.SL: "SL",
        OrderType.SL_M: "SL-M",
    }

    def __init__(self):
        """Initialize Upstox broker."""
        self.api_key = os.getenv("UPSTOX_API_KEY", "")
        self.api_secret = os.getenv("UPSTOX_API_SECRET", "")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")

        self._connected = False
        self._session = requests.Session()

        # Rate limiter (conservative: 200 RPM, 15 RPS)
        self._limiter = BrokerRateLimiter("upstox")

        # Circuit breaker for API protection
        self._circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # Token storage path
        self._token_path = Path(__file__).parent.parent / "data" / "upstox_token.json"

        # NSE holidays (update annually)
        self._holidays = self._load_holidays()

        # Fund cache for resilience against API failures
        self._last_known_funds: Optional[Funds] = None
        self._funds_failure_count: int = 0

    def _load_holidays(self) -> List[date]:
        """Load NSE holidays."""
        return [
            date(2026, 1, 26),   # Republic Day
            date(2026, 3, 10),   # Maha Shivaratri
            date(2026, 3, 17),   # Holi
            date(2026, 4, 2),    # Ram Navami
            date(2026, 4, 6),    # Mahavir Jayanti
            date(2026, 4, 10),   # Good Friday
            date(2026, 4, 14),   # Ambedkar Jayanti
            date(2026, 5, 1),    # May Day
            date(2026, 8, 15),   # Independence Day
            date(2026, 8, 19),   # Muharram
            date(2026, 10, 2),   # Gandhi Jayanti
            date(2026, 10, 21),  # Dussehra
            date(2026, 10, 28),  # Milad-un-Nabi
            date(2026, 11, 4),   # Diwali (Laxmi Puja)
            date(2026, 11, 5),   # Diwali (Balipratipada)
            date(2026, 11, 16),  # Guru Nanak Jayanti
            date(2026, 12, 25),  # Christmas
        ]

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        retry: bool = True
    ) -> Dict[str, Any]:
        """
        Make rate-limited API request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body

        Returns:
            Response JSON
        """
        url = f"{self.BASE_URL}{endpoint}"

        # Rate limit
        self._limiter.wait(endpoint)

        # Circuit breaker check
        if not self._circuit.allow_request():
            raise BrokerException("Circuit breaker open - API unavailable")

        try:
            response = self._session.request(
                method=method,
                url=url,
                headers=self._get_headers(),
                params=params,
                json=data,
                timeout=30
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                self._limiter.record_failure(endpoint, is_rate_limit=True)
                raise RateLimitException(
                    f"Rate limited on {endpoint}",
                    retry_after=retry_after
                )

            # Handle auth errors
            if response.status_code in (401, 403):
                raise AuthException(f"Authentication failed: {response.text}")

            # Handle other errors
            if response.status_code >= 400:
                self._circuit.record_failure()
                raise BrokerException(f"API error {response.status_code}: {response.text}")

            self._circuit.record_success()
            self._limiter.record_success(endpoint)

            return response.json()

        except requests.exceptions.Timeout:
            self._circuit.record_failure()
            raise BrokerException("Request timeout")

        except requests.exceptions.ConnectionError as e:
            self._circuit.record_failure()
            raise BrokerException(f"Connection error: {e}")

    def _format_symbol(self, symbol: str) -> str:
        """
        Format symbol for Upstox API.

        Upstox uses format: NSE_EQ|SYMBOL
        """
        if "|" in symbol:
            return symbol
        return f"NSE_EQ|{symbol}"

    def _parse_symbol(self, upstox_symbol: str) -> str:
        """Parse Upstox symbol to simple format."""
        if "|" in upstox_symbol:
            return upstox_symbol.split("|")[1]
        return upstox_symbol

    # === CONNECTION ===

    def connect(self) -> bool:
        """
        Connect to Upstox API.

        Validates access token by making a test request.
        """
        if not self.access_token:
            logger.error("UPSTOX_ACCESS_TOKEN not set in environment")
            return False

        try:
            # Test connection by getting profile
            response = self._make_request("GET", "/user/profile")

            if response.get("status") == "success":
                user = response.get("data", {})
                logger.info(f"Connected to Upstox as {user.get('user_name', 'Unknown')}")
                self._connected = True
                return True
            else:
                logger.error(f"Connection failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def disconnect(self):
        """Disconnect from Upstox."""
        self._session.close()
        self._connected = False
        logger.info("Disconnected from Upstox")

    # === MARKET DATA ===

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_ltp(self, symbol: str) -> float:
        """Get Last Traded Price with sanity validation."""
        formatted = self._format_symbol(symbol)

        response = self._make_request(
            "GET",
            "/market-quote/ltp",
            params={"instrument_key": formatted}
        )

        if response.get("status") == "success":
            data = response.get("data", {}).get(formatted, {})
            price = float(data.get("last_price", 0))
            # Sanity check: reject invalid prices
            if price <= 0 or price > 1_000_000:
                logger.warning(f"Invalid LTP for {symbol}: {price}")
                return 0.0
            import math
            if math.isnan(price) or math.isinf(price):
                logger.warning(f"NaN/Inf LTP for {symbol}")
                return 0.0
            return price

        return 0.0

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get full market quote."""
        formatted = self._format_symbol(symbol)

        response = self._make_request(
            "GET",
            "/market-quote/quotes",
            params={"instrument_key": formatted}
        )

        if response.get("status") != "success":
            return None

        data = response.get("data", {}).get(formatted, {})
        ohlc = data.get("ohlc", {})

        return Quote(
            symbol=symbol,
            ltp=float(data.get("last_price", 0)),
            open=float(ohlc.get("open", 0)),
            high=float(ohlc.get("high", 0)),
            low=float(ohlc.get("low", 0)),
            close=float(ohlc.get("close", 0)),
            volume=int(data.get("volume", 0)),
            bid=float(data.get("depth", {}).get("buy", [{}])[0].get("price", 0)),
            ask=float(data.get("depth", {}).get("sell", [{}])[0].get("price", 0)),
            timestamp=now_ist().replace(tzinfo=None)
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        quotes = {}

        # Upstox allows batch requests
        formatted = [self._format_symbol(s) for s in symbols]
        instrument_keys = ",".join(formatted)

        try:
            response = self._make_request(
                "GET",
                "/market-quote/quotes",
                params={"instrument_key": instrument_keys}
            )

            if response.get("status") == "success":
                for key, data in response.get("data", {}).items():
                    symbol = self._parse_symbol(key)
                    ohlc = data.get("ohlc", {})

                    quotes[symbol] = Quote(
                        symbol=symbol,
                        ltp=float(data.get("last_price", 0)),
                        open=float(ohlc.get("open", 0)),
                        high=float(ohlc.get("high", 0)),
                        low=float(ohlc.get("low", 0)),
                        close=float(ohlc.get("close", 0)),
                        volume=int(data.get("volume", 0)),
                        bid=float(data.get("depth", {}).get("buy", [{}])[0].get("price", 0)),
                        ask=float(data.get("depth", {}).get("sell", [{}])[0].get("price", 0)),
                        timestamp=now_ist().replace(tzinfo=None)
                    )

        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")

        return quotes

    # === ORDER MANAGEMENT ===

    # NO retry on place_order - retrying can cause DOUBLE ORDERS on timeout
    def place_order(self, order: Order) -> OrderResponse:
        """Place an order on Upstox."""
        now = now_ist().replace(tzinfo=None)

        payload = {
            "quantity": order.quantity,
            "product": self.PRODUCT_MAP.get(order.product, "I"),
            "validity": "DAY",
            "price": order.price or 0,
            "tag": order.tag or "",
            "instrument_token": self._format_symbol(order.symbol),
            "order_type": self.ORDER_TYPE_MAP.get(order.order_type, "MARKET"),
            "transaction_type": order.side.value,
            "disclosed_quantity": 0,
            "trigger_price": order.trigger_price or 0,
            "is_amo": False
        }

        try:
            response = self._make_request("POST", "/order/place", data=payload)

            if response.get("status") == "success":
                order_id = response.get("data", {}).get("order_id", "")
                return OrderResponse(
                    order_id=order_id,
                    status=OrderStatus.PENDING,
                    message="Order placed successfully",
                    timestamp=now
                )
            else:
                return OrderResponse(
                    order_id="",
                    status=OrderStatus.REJECTED,
                    message=response.get("message", "Order rejected"),
                    timestamp=now
                )

        except Exception as e:
            return OrderResponse(
                order_id="",
                status=OrderStatus.REJECTED,
                message=str(e),
                timestamp=now
            )

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        quantity: Optional[int] = None
    ) -> bool:
        """Modify an existing order."""
        payload = {"order_id": order_id}

        if price is not None:
            payload["price"] = price
        if trigger_price is not None:
            payload["trigger_price"] = trigger_price
        if quantity is not None:
            payload["quantity"] = quantity

        try:
            response = self._make_request("PUT", "/order/modify", data=payload)
            return response.get("status") == "success"
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            response = self._make_request(
                "DELETE",
                f"/order/cancel",
                params={"order_id": order_id}
            )
            return response.get("status") == "success"
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderBook]:
        """Get status of a specific order."""
        try:
            response = self._make_request(
                "GET",
                "/order/details",
                params={"order_id": order_id}
            )

            if response.get("status") != "success":
                return None

            data = response.get("data", {})
            return self._parse_order(data)

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None

    def get_order_book(self) -> List[OrderBook]:
        """Get all orders for today."""
        try:
            response = self._make_request("GET", "/order/retrieve-all")

            if response.get("status") != "success":
                return []

            orders = []
            for order_data in response.get("data", []):
                order = self._parse_order(order_data)
                if order:
                    orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return []

    def _parse_order(self, data: Dict) -> Optional[OrderBook]:
        """Parse Upstox order data to OrderBook."""
        try:
            status_map = {
                "complete": OrderStatus.COMPLETE,
                "rejected": OrderStatus.REJECTED,
                "cancelled": OrderStatus.CANCELLED,
                "open": OrderStatus.OPEN,
                "pending": OrderStatus.PENDING,
                "partial_fill": OrderStatus.PARTIAL_FILL,
                "partially_filled": OrderStatus.PARTIAL_FILL,
                "trigger_pending": OrderStatus.OPEN,
                "not_cancelled": OrderStatus.OPEN,
                "expired": OrderStatus.CANCELLED,
            }

            status_str = data.get("status", "").lower()
            status = status_map.get(status_str, OrderStatus.PENDING)

            # Warn on unknown status
            if status_str and status_str not in status_map:
                logger.warning(f"Unknown order status from Upstox: '{status_str}', defaulting to PENDING")

            # Alert on partial fills
            filled = int(data.get("filled_quantity", 0))
            pending = int(data.get("pending_quantity", 0))
            if filled > 0 and pending > 0:
                logger.warning(
                    f"PARTIAL FILL: {data.get('instrument_token', '')} "
                    f"filled={filled}, pending={pending}, status={status_str}"
                )

            return OrderBook(
                order_id=data.get("order_id", ""),
                symbol=self._parse_symbol(data.get("instrument_token", "")),
                side=OrderSide(data.get("transaction_type", "BUY")),
                quantity=int(data.get("quantity", 0)),
                filled_quantity=int(data.get("filled_quantity", 0)),
                pending_quantity=int(data.get("pending_quantity", 0)),
                order_type=OrderType(data.get("order_type", "MARKET")),
                price=float(data.get("price", 0)),
                trigger_price=float(data.get("trigger_price", 0)),
                average_price=float(data.get("average_price", 0)),
                status=status,
                placed_at=datetime.fromisoformat(data.get("order_timestamp", now_ist().replace(tzinfo=None).isoformat())),
                updated_at=now_ist().replace(tzinfo=None)
            )
        except Exception as e:
            logger.warning(f"Failed to parse order: {e}")
            return None

    # === POSITIONS ===

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            response = self._make_request("GET", "/portfolio/short-term-positions")

            if response.get("status") != "success":
                return []

            positions = []
            for pos_data in response.get("data", []):
                pos = self._parse_position(pos_data)
                if pos and pos.quantity != 0:
                    positions.append(pos)

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    def _parse_position(self, data: Dict) -> Optional[Position]:
        """Parse Upstox position data."""
        try:
            quantity = int(data.get("quantity", 0))
            avg_price = float(data.get("average_price", 0))
            ltp = float(data.get("last_price", 0))

            pnl = (ltp - avg_price) * quantity if quantity > 0 else 0
            pnl_pct = ((ltp / avg_price) - 1) * 100 if avg_price > 0 else 0

            return Position(
                symbol=self._parse_symbol(data.get("instrument_token", "")),
                quantity=quantity,
                average_price=avg_price,
                last_price=ltp,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                product=ProductType.INTRADAY,
                value=round(quantity * ltp, 2)
            )
        except Exception as e:
            logger.warning(f"Failed to parse position: {e}")
            return None

    # === ACCOUNT ===

    def get_funds(self) -> Funds:
        """Get available funds with cache fallback on API failure."""
        try:
            response = self._make_request("GET", "/user/get-funds-and-margin")

            if response.get("status") != "success":
                logger.warning("Funds API returned non-success status")
                return self._get_cached_funds_or_zero()

            # Upstox returns equity segment data
            equity = response.get("data", {}).get("equity", {})

            available = float(equity.get("available_margin", 0))
            used = float(equity.get("used_margin", 0))

            funds = Funds(
                available_cash=round(available, 2),
                used_margin=round(used, 2),
                total_balance=round(available + used, 2)
            )
            # Cache successful response
            self._last_known_funds = funds
            self._funds_failure_count = 0
            return funds

        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return self._get_cached_funds_or_zero()

    def _get_cached_funds_or_zero(self) -> Funds:
        """Return cached funds on API failure. Uses cache permanently to avoid blocking trading."""
        self._funds_failure_count += 1
        if self._last_known_funds:
            if self._funds_failure_count >= 3:
                logger.error(
                    f"Funds API failed {self._funds_failure_count} consecutive times. "
                    f"Using STALE cached funds: Rs.{self._last_known_funds.available_cash:.0f}. "
                    f"Check broker connectivity!"
                )
                from utils.alerts import alert_error
                alert_error(
                    "Funds API unreachable",
                    f"Failed {self._funds_failure_count}x. Using cached Rs.{self._last_known_funds.available_cash:.0f}. "
                    f"Verify broker dashboard."
                )
            else:
                logger.warning(
                    f"Using cached funds (failure #{self._funds_failure_count}): "
                    f"available={self._last_known_funds.available_cash}"
                )
            return self._last_known_funds
        # No cache at all — truly unknown funds, return zero to prevent blind trading
        logger.error(f"Funds API failed {self._funds_failure_count}x with NO cached data — returning zero")
        return Funds(available_cash=0, used_margin=0, total_balance=0)

    def get_pnl(self) -> PnL:
        """Get today's P&L by matching buy/sell trades per symbol."""
        try:
            # Get positions for unrealized
            positions = self.get_positions()
            unrealized = sum(p.pnl for p in positions)

            # Get trade book for realized - match buys to sells per symbol
            response = self._make_request("GET", "/order/trades/get-trades-for-day")
            realized = 0

            if response.get("status") == "success":
                # Group trades by symbol
                symbol_trades: dict = {}
                for trade in response.get("data", []):
                    sym = self._parse_symbol(trade.get("instrument_token", ""))
                    if sym not in symbol_trades:
                        symbol_trades[sym] = {"buys": [], "sells": []}
                    side = trade.get("transaction_type", "").upper()
                    qty = int(trade.get("quantity", 0))
                    price = float(trade.get("average_price", 0))
                    if side == "BUY":
                        symbol_trades[sym]["buys"].append((qty, price))
                    elif side == "SELL":
                        symbol_trades[sym]["sells"].append((qty, price))

                # Calculate realized P&L per symbol (proper FIFO matching)
                for sym, trades in symbol_trades.items():
                    # Build FIFO buy queue
                    buy_queue = list(trades["buys"])  # [(qty, price), ...]
                    buy_idx = 0
                    buy_remaining = buy_queue[0][0] if buy_queue else 0

                    for sell_qty, sell_price in trades["sells"]:
                        remaining_sell = sell_qty
                        while remaining_sell > 0 and buy_idx < len(buy_queue):
                            match_qty = min(remaining_sell, buy_remaining)
                            buy_price = buy_queue[buy_idx][1]
                            realized += (sell_price - buy_price) * match_qty
                            remaining_sell -= match_qty
                            buy_remaining -= match_qty
                            if buy_remaining <= 0:
                                buy_idx += 1
                                if buy_idx < len(buy_queue):
                                    buy_remaining = buy_queue[buy_idx][0]
                        if remaining_sell > 0:
                            logger.warning(f"FIFO: {sym} sold {remaining_sell} shares without matching buy")

            return PnL(
                realized=round(realized, 2),
                unrealized=round(unrealized, 2),
                total=round(realized + unrealized, 2)
            )

        except Exception as e:
            logger.error(f"Failed to get P&L: {e}")
            return PnL(realized=0, unrealized=0, total=0)

    # === UTILITIES ===

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = now_ist().replace(tzinfo=None)

        # Check weekend
        if now.weekday() >= 5:
            return False

        # Check holidays
        if now.date() in self._holidays:
            return False

        # Check market hours (9:15 AM to 3:30 PM)
        market_start = dt_time(9, 15)
        market_end = dt_time(15, 30)

        return market_start <= now.time() <= market_end

    def get_holidays(self) -> List[date]:
        """Get market holidays."""
        return self._holidays.copy()

    def check_token_valid(self) -> bool:
        """Check if the access token is still valid by hitting the profile endpoint."""
        try:
            response = self._make_request("GET", "/user/profile")
            return response.get("status") == "success"
        except AuthException:
            return False
        except Exception:
            # Network errors etc - don't treat as token failure
            return True

    def get_next_trading_day(self) -> date:
        """Get the next trading day."""
        check_date = today_ist()

        for _ in range(10):  # Check up to 10 days ahead
            check_date += timedelta(days=1)

            # Skip weekends
            if check_date.weekday() >= 5:
                continue

            # Skip holidays
            if check_date in self._holidays:
                continue

            return check_date

        return check_date
