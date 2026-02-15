"""
Zerodha Broker - Kite Connect API integration.

Uses the official kiteconnect SDK for all API calls.
Mirrors the same safety patterns as UpstoxBroker:
- NO retry on order placement (prevents double orders)
- Circuit breaker for API protection
- Rate limiting on all endpoints
- Proper error categorization (auth vs transient vs fatal)
"""
import os
import json
from datetime import datetime, date, time as dt_time
from typing import List, Optional, Dict
from pathlib import Path
from loguru import logger

from kiteconnect import KiteConnect
from kiteconnect import exceptions as kite_exc

from broker.base import (
    BaseBroker, Order, OrderSide, OrderType, OrderStatus,
    OrderResponse, OrderBook, Position, Quote, Funds, PnL, ProductType
)
from utils.rate_limiter import BrokerRateLimiter
from utils.error_handler import (
    BrokerException, RateLimitException, AuthException,
    retry_with_backoff, CircuitBreaker
)
from utils.platform import now_ist, today_ist


class ZerodhaBroker(BaseBroker):
    """
    Zerodha Kite Connect broker implementation.

    Requires environment variables:
    - ZERODHA_API_KEY: Kite Connect API key
    - ZERODHA_API_SECRET: Kite Connect API secret
    - ZERODHA_ACCESS_TOKEN: Daily access token (auto-refreshed by zerodha_auto_login.py)
    """

    # Product type mapping: internal → Kite
    PRODUCT_MAP = {
        ProductType.INTRADAY: "MIS",
        ProductType.DELIVERY: "CNC",
        ProductType.CNC: "CNC",
    }

    # Order type mapping: internal → Kite
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "MARKET",
        OrderType.LIMIT: "LIMIT",
        OrderType.SL: "SL",
        OrderType.SL_M: "SL-M",
    }

    # Kite order status → internal OrderStatus
    STATUS_MAP = {
        "COMPLETE": OrderStatus.COMPLETE,
        "REJECTED": OrderStatus.REJECTED,
        "CANCELLED": OrderStatus.CANCELLED,
        "OPEN": OrderStatus.OPEN,
        "PENDING": OrderStatus.PENDING,
        "TRIGGER PENDING": OrderStatus.OPEN,  # SL order waiting for trigger
    }

    def __init__(self):
        self.api_key = os.getenv("ZERODHA_API_KEY", "")
        self.api_secret = os.getenv("ZERODHA_API_SECRET", "")
        self.access_token = os.getenv("ZERODHA_ACCESS_TOKEN", "")

        # Try loading token from file if env is empty
        if not self.access_token:
            self.access_token = self._load_token_from_file()

        # Initialize Kite Connect SDK
        self._kite = KiteConnect(api_key=self.api_key)
        if self.access_token:
            self._kite.set_access_token(self.access_token)

        self._connected = False

        # Rate limiter: Zerodha allows 10 RPS global, but we stay conservative
        self._limiter = BrokerRateLimiter("zerodha")

        # Circuit breaker for API protection
        self._circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # NSE holidays
        self._holidays = self._load_holidays()

        # Fund cache for resilience against API failures
        self._last_known_funds: Optional[Funds] = None
        self._funds_failure_count: int = 0

    def _load_token_from_file(self) -> str:
        """Load access token from saved file."""
        token_path = Path(__file__).parent.parent / "data" / "zerodha_token.json"
        try:
            if token_path.exists():
                data = json.loads(token_path.read_text())
                token = data.get("access_token", "")
                if token:
                    logger.debug("Loaded Zerodha token from file")
                    return token
        except Exception as e:
            logger.debug(f"Could not load token from file: {e}")
        return ""

    def _load_holidays(self) -> List[date]:
        """NSE holidays 2026 (update annually)."""
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

    def _api_call(self, func, *args, **kwargs):
        """
        Central wrapper for all Kite SDK calls.

        Handles: rate limiting, circuit breaker, error mapping.
        Every SDK call goes through this.
        """
        endpoint = func.__name__

        self._limiter.wait(endpoint)

        if not self._circuit.allow_request():
            raise BrokerException("Circuit breaker open - Zerodha API unavailable")

        try:
            result = func(*args, **kwargs)
            self._circuit.record_success()
            self._limiter.record_success(endpoint)
            return result

        except kite_exc.TokenException as e:
            raise AuthException(f"Zerodha token expired/invalid: {e}")

        except kite_exc.PermissionException as e:
            raise AuthException(f"Zerodha permission denied: {e}")

        except kite_exc.OrderException as e:
            self._circuit.record_failure()
            raise BrokerException(f"Zerodha order error: {e}")

        except kite_exc.InputException as e:
            raise BrokerException(f"Zerodha input error: {e}")

        except kite_exc.DataException as e:
            self._circuit.record_failure()
            raise BrokerException(f"Zerodha data error: {e}")

        except kite_exc.NetworkException as e:
            self._circuit.record_failure()
            raise BrokerException(f"Zerodha network error: {e}")

        except kite_exc.GeneralException as e:
            self._circuit.record_failure()
            raise BrokerException(f"Zerodha error: {e}")

    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Kite API: RELIANCE → NSE:RELIANCE"""
        if ":" in symbol:
            return symbol
        return f"NSE:{symbol}"

    def _parse_symbol(self, kite_symbol: str) -> str:
        """Parse Kite symbol to simple format: NSE:RELIANCE → RELIANCE"""
        if ":" in kite_symbol:
            return kite_symbol.split(":")[1]
        return kite_symbol

    # === CONNECTION ===

    def connect(self) -> bool:
        if not self.access_token:
            logger.error("ZERODHA_ACCESS_TOKEN not set")
            return False

        try:
            profile = self._api_call(self._kite.profile)
            user_name = profile.get("user_name", "Unknown")
            logger.info(f"Connected to Zerodha as {user_name}")
            self._connected = True
            return True
        except AuthException:
            logger.error("Zerodha token expired. Run: python scripts/zerodha_auto_login.py")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha: {e}")
            return False

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self):
        self._connected = False
        logger.info("Disconnected from Zerodha")

    # === MARKET DATA ===

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_ltp(self, symbol: str) -> float:
        """Get Last Traded Price with sanity validation."""
        import math
        formatted = self._format_symbol(symbol)
        data = self._api_call(self._kite.ltp, [formatted])
        entry = data.get(formatted, {})
        price = float(entry.get("last_price", 0))
        # Sanity check: reject invalid prices
        if price <= 0 or price > 1_000_000 or math.isnan(price) or math.isinf(price):
            logger.warning(f"Invalid LTP for {symbol}: {price}")
            return 0.0
        return price

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def get_quote(self, symbol: str) -> Optional[Quote]:
        formatted = self._format_symbol(symbol)
        data = self._api_call(self._kite.quote, [formatted])

        entry = data.get(formatted)
        if not entry:
            return None

        ohlc = entry.get("ohlc", {})
        depth = entry.get("depth", {})
        buy_depth = depth.get("buy", [{}])
        sell_depth = depth.get("sell", [{}])

        return Quote(
            symbol=symbol,
            ltp=float(entry.get("last_price", 0)),
            open=float(ohlc.get("open", 0)),
            high=float(ohlc.get("high", 0)),
            low=float(ohlc.get("low", 0)),
            close=float(ohlc.get("close", 0)),
            volume=int(entry.get("volume", 0)),
            bid=float(buy_depth[0].get("price", 0)) if buy_depth else 0,
            ask=float(sell_depth[0].get("price", 0)) if sell_depth else 0,
            timestamp=now_ist().replace(tzinfo=None)
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        formatted_list = [self._format_symbol(s) for s in symbols]

        try:
            data = self._api_call(self._kite.quote, formatted_list)
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}

        quotes = {}
        for formatted_key, entry in data.items():
            symbol = self._parse_symbol(formatted_key)
            ohlc = entry.get("ohlc", {})
            depth = entry.get("depth", {})
            buy_depth = depth.get("buy", [{}])
            sell_depth = depth.get("sell", [{}])

            quotes[symbol] = Quote(
                symbol=symbol,
                ltp=float(entry.get("last_price", 0)),
                open=float(ohlc.get("open", 0)),
                high=float(ohlc.get("high", 0)),
                low=float(ohlc.get("low", 0)),
                close=float(ohlc.get("close", 0)),
                volume=int(entry.get("volume", 0)),
                bid=float(buy_depth[0].get("price", 0)) if buy_depth else 0,
                ask=float(sell_depth[0].get("price", 0)) if sell_depth else 0,
                timestamp=now_ist().replace(tzinfo=None)
            )

        return quotes

    # === ORDER MANAGEMENT ===

    # NO retry on place_order — retrying can cause DOUBLE ORDERS on timeout
    def place_order(self, order: Order) -> OrderResponse:
        now = now_ist().replace(tzinfo=None)

        try:
            order_id = self._api_call(
                self._kite.place_order,
                variety=self._kite.VARIETY_REGULAR,
                exchange=self._kite.EXCHANGE_NSE,
                tradingsymbol=order.symbol,
                transaction_type=order.side.value,
                quantity=order.quantity,
                product=self.PRODUCT_MAP.get(order.product, "MIS"),
                order_type=self.ORDER_TYPE_MAP.get(order.order_type, "MARKET"),
                price=order.price or 0,
                trigger_price=order.trigger_price or 0,
                validity=self._kite.VALIDITY_DAY,
                tag=order.tag or ""
            )

            # Kite returns just the order_id on success
            return OrderResponse(
                order_id=str(order_id),
                status=OrderStatus.PENDING,
                message="Order placed successfully",
                timestamp=now
            )

        except Exception as e:
            logger.error(f"Order placement failed for {order.symbol}: {e}")
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
        kwargs = {
            "variety": self._kite.VARIETY_REGULAR,
            "order_id": order_id,
        }
        if price is not None:
            kwargs["price"] = price
        if trigger_price is not None:
            kwargs["trigger_price"] = trigger_price
        if quantity is not None:
            kwargs["quantity"] = quantity

        try:
            self._api_call(self._kite.modify_order, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._api_call(
                self._kite.cancel_order,
                variety=self._kite.VARIETY_REGULAR,
                order_id=order_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[OrderBook]:
        try:
            history = self._api_call(self._kite.order_history, order_id=order_id)
            if not history:
                return None
            # Last entry has the latest status
            return self._parse_order(history[-1])
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    def get_order_book(self) -> List[OrderBook]:
        try:
            orders_data = self._api_call(self._kite.orders)
            if not orders_data:
                return []

            orders = []
            for entry in orders_data:
                parsed = self._parse_order(entry)
                if parsed:
                    orders.append(parsed)
            return orders

        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return []

    def _parse_order(self, data: Dict) -> Optional[OrderBook]:
        """Parse a single Kite order dict into OrderBook."""
        try:
            status_str = (data.get("status") or "").upper()
            status = self.STATUS_MAP.get(status_str, OrderStatus.PENDING)

            # Warn on unknown status
            if status_str and status_str not in self.STATUS_MAP:
                logger.warning(f"Unknown order status from Zerodha: '{status_str}', defaulting to PENDING")

            quantity = int(data.get("quantity", 0))
            filled = int(data.get("filled_quantity", 0))
            pending = int(data.get("pending_quantity", quantity - filled))

            # Alert on partial fills
            if filled > 0 and pending > 0:
                logger.warning(
                    f"PARTIAL FILL: {data.get('tradingsymbol', '')} "
                    f"filled={filled}, pending={pending}, status={status_str}"
                )

            # Map order type string back to enum
            ot_str = (data.get("order_type") or "MARKET").upper()
            order_type_map = {
                "MARKET": OrderType.MARKET,
                "LIMIT": OrderType.LIMIT,
                "SL": OrderType.SL,
                "SL-M": OrderType.SL_M,
            }

            # Map side string back to enum
            side_str = (data.get("transaction_type") or "BUY").upper()

            placed_str = data.get("order_timestamp") or data.get("exchange_timestamp")
            if placed_str:
                try:
                    placed_at = datetime.fromisoformat(str(placed_str))
                except (ValueError, TypeError):
                    placed_at = now_ist().replace(tzinfo=None)
            else:
                placed_at = now_ist().replace(tzinfo=None)

            return OrderBook(
                order_id=str(data.get("order_id", "")),
                symbol=data.get("tradingsymbol", ""),
                side=OrderSide(side_str) if side_str in ("BUY", "SELL") else OrderSide.BUY,
                quantity=quantity,
                filled_quantity=filled,
                pending_quantity=pending,
                order_type=order_type_map.get(ot_str, OrderType.MARKET),
                price=float(data.get("price", 0) or 0),
                trigger_price=float(data.get("trigger_price", 0) or 0),
                average_price=float(data.get("average_price", 0) or 0),
                status=status,
                placed_at=placed_at,
                updated_at=now_ist().replace(tzinfo=None)
            )
        except Exception as e:
            logger.warning(f"Failed to parse order: {e}")
            return None

    # === POSITIONS ===

    def get_positions(self) -> List[Position]:
        try:
            pos_data = self._api_call(self._kite.positions)
            net_positions = pos_data.get("net", [])

            positions = []
            for entry in net_positions:
                parsed = self._parse_position(entry)
                if parsed and parsed.quantity != 0:
                    positions.append(parsed)
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def _parse_position(self, data: Dict) -> Optional[Position]:
        """Parse a single Kite position dict into Position."""
        try:
            quantity = int(data.get("quantity", 0))
            avg_price = float(data.get("average_price", 0) or 0)
            ltp = float(data.get("last_price", 0) or 0)

            if quantity != 0 and avg_price > 0:
                pnl = (ltp - avg_price) * quantity
                pnl_pct = ((ltp / avg_price) - 1) * 100
            else:
                pnl = 0
                pnl_pct = 0

            # Map product type
            product_str = (data.get("product") or "MIS").upper()
            product_map = {"MIS": ProductType.INTRADAY, "CNC": ProductType.CNC}
            product = product_map.get(product_str, ProductType.INTRADAY)

            return Position(
                symbol=data.get("tradingsymbol", ""),
                quantity=abs(quantity),
                average_price=avg_price,
                last_price=ltp,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                product=product,
                value=round(abs(quantity) * ltp, 2)
            )
        except Exception as e:
            logger.warning(f"Failed to parse position: {e}")
            return None

    # === ACCOUNT ===

    def get_funds(self) -> Funds:
        """Get available funds with cache fallback on API failure."""
        try:
            margins = self._api_call(self._kite.margins, segment="equity")

            available = float(margins.get("available", {}).get("live_balance", 0) or 0)
            used = float(margins.get("utilised", {}).get("debits", 0) or 0)

            # Fallback: try net field
            if available == 0:
                available = float(margins.get("net", 0) or 0) - used

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
        try:
            # Unrealized from open positions
            positions = self.get_positions()
            unrealized = sum(p.pnl for p in positions)

            # Realized from today's closed trades
            realized = 0.0
            try:
                pos_data = self._api_call(self._kite.positions)
                for entry in pos_data.get("day", []):
                    qty = int(entry.get("quantity", 0))
                    if qty == 0:
                        # Position fully closed today — m2m is realized
                        realized += float(entry.get("m2m", 0) or 0)
            except Exception:
                pass

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
        now = now_ist().replace(tzinfo=None)

        if now.weekday() >= 5:
            return False

        if now.date() in self._holidays:
            return False

        market_start = dt_time(9, 15)
        market_end = dt_time(15, 30)
        return market_start <= now.time() <= market_end

    def get_holidays(self) -> List[date]:
        return self._holidays.copy()

    def check_token_valid(self) -> bool:
        """Check if the access token is still valid."""
        try:
            self._api_call(self._kite.profile)
            return True
        except AuthException:
            return False
        except Exception:
            # Network errors — don't treat as token failure
            return True
