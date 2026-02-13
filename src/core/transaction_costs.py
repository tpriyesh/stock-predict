"""
TransactionCosts - Realistic Indian Market Cost Modeling

CRITICAL: Many backtested edges DISAPPEAR after transaction costs.

This module models ALL costs in Indian equity trading:
1. Brokerage (varies by broker)
2. Securities Transaction Tax (STT)
3. Exchange transaction charges (NSE/BSE)
4. GST on brokerage
5. Stamp duty
6. SEBI turnover fee
7. Slippage (bid-ask spread, market impact)

Without proper cost modeling, you might think you have an edge when you don't.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class TradingSegment(Enum):
    """Trading segment affects costs."""
    EQUITY_DELIVERY = "EQUITY_DELIVERY"      # CNC - Delivery trading
    EQUITY_INTRADAY = "EQUITY_INTRADAY"      # MIS - Intraday
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class BrokerType(Enum):
    """Broker type affects brokerage."""
    DISCOUNT = "DISCOUNT"        # Zerodha, Groww, Upstox
    FULL_SERVICE = "FULL_SERVICE"  # ICICI Direct, HDFC Securities
    ULTRA_LOW = "ULTRA_LOW"      # Finvasia (zero brokerage)


@dataclass
class CostBreakdown:
    """Detailed breakdown of transaction costs."""
    # Absolute values (in INR)
    brokerage: float
    stt: float
    exchange_txn: float
    gst: float
    stamp_duty: float
    sebi_fee: float
    slippage: float

    # Totals
    total_buy_cost: float
    total_sell_cost: float
    total_round_trip: float

    # As percentage of trade value
    total_pct: float

    def to_dict(self) -> dict:
        return {
            'brokerage': round(self.brokerage, 2),
            'stt': round(self.stt, 2),
            'exchange_txn': round(self.exchange_txn, 2),
            'gst': round(self.gst, 2),
            'stamp_duty': round(self.stamp_duty, 2),
            'sebi_fee': round(self.sebi_fee, 2),
            'slippage': round(self.slippage, 2),
            'total_buy_cost': round(self.total_buy_cost, 2),
            'total_sell_cost': round(self.total_sell_cost, 2),
            'total_round_trip': round(self.total_round_trip, 2),
            'total_pct': round(self.total_pct * 100, 4)
        }


@dataclass
class TransactionCosts:
    """
    Indian market transaction costs.

    All percentages are expressed as decimals (e.g., 0.001 = 0.1%)

    Default values are for discount brokers as of 2024.
    """

    # Brokerage (varies by broker and segment)
    brokerage_pct: float = 0.0003          # 0.03% or Rs 20 flat (whichever lower)
    brokerage_flat_max: float = 20.0       # Rs 20 cap for discount brokers

    # Securities Transaction Tax (STT) - Government mandated
    stt_buy_delivery: float = 0.001        # 0.1% on buy (delivery)
    stt_sell_delivery: float = 0.001       # 0.1% on sell (delivery)
    stt_sell_intraday: float = 0.00025     # 0.025% on sell only (intraday)

    # Exchange Transaction Charges
    nse_txn_pct: float = 0.0000297         # 0.00297% NSE
    bse_txn_pct: float = 0.000030          # 0.003% BSE

    # GST on brokerage and exchange charges
    gst_pct: float = 0.18                  # 18% GST

    # Stamp Duty (varies by state, using highest)
    stamp_duty_pct: float = 0.00015        # 0.015% on buy side

    # SEBI Turnover Fee
    sebi_fee_pct: float = 0.000001         # 0.0001%

    # Slippage (bid-ask spread + market impact)
    slippage_pct: float = 0.001            # 0.1% estimated slippage
    slippage_volume_factor: float = 0.5    # Higher for larger orders

    # Settings
    segment: TradingSegment = TradingSegment.EQUITY_DELIVERY
    broker_type: BrokerType = BrokerType.DISCOUNT
    exchange: str = "NSE"

    def __post_init__(self):
        """Adjust costs based on broker type."""
        if self.broker_type == BrokerType.FULL_SERVICE:
            self.brokerage_pct = 0.005  # 0.5% for full service
            self.brokerage_flat_max = float('inf')  # No cap
        elif self.broker_type == BrokerType.ULTRA_LOW:
            self.brokerage_pct = 0.0
            self.brokerage_flat_max = 0.0

    def calculate_brokerage(self, trade_value: float) -> float:
        """Calculate brokerage with flat cap."""
        pct_brokerage = trade_value * self.brokerage_pct
        return min(pct_brokerage, self.brokerage_flat_max)

    def calculate_stt(self, trade_value: float, is_buy: bool) -> float:
        """Calculate STT based on segment and direction."""
        if self.segment == TradingSegment.EQUITY_DELIVERY:
            # STT on both buy and sell for delivery
            return trade_value * (self.stt_buy_delivery if is_buy else self.stt_sell_delivery)
        elif self.segment == TradingSegment.EQUITY_INTRADAY:
            # STT only on sell for intraday
            return 0 if is_buy else trade_value * self.stt_sell_intraday
        else:
            # For F&O, different rates apply
            return trade_value * 0.0001  # Simplified

    def calculate_exchange_charges(self, trade_value: float) -> float:
        """Calculate exchange transaction charges."""
        if self.exchange == "NSE":
            return trade_value * self.nse_txn_pct
        else:
            return trade_value * self.bse_txn_pct

    def calculate_gst(self, brokerage: float, exchange_charges: float) -> float:
        """GST is charged on brokerage + exchange charges."""
        return (brokerage + exchange_charges) * self.gst_pct

    def calculate_stamp_duty(self, trade_value: float, is_buy: bool) -> float:
        """Stamp duty only on buy side."""
        return trade_value * self.stamp_duty_pct if is_buy else 0

    def calculate_sebi_fee(self, trade_value: float) -> float:
        """SEBI turnover fee."""
        return trade_value * self.sebi_fee_pct

    def calculate_slippage(self,
                           trade_value: float,
                           avg_volume: float = 1_000_000,
                           order_size: float = 0) -> float:
        """
        Estimate slippage based on order size relative to volume.

        Larger orders relative to daily volume = more slippage.
        """
        base_slippage = trade_value * self.slippage_pct

        # Adjust for order size relative to volume
        if order_size > 0 and avg_volume > 0:
            volume_impact = (order_size / avg_volume) * self.slippage_volume_factor
            volume_impact = min(volume_impact, 0.02)  # Cap at 2% extra
            base_slippage += trade_value * volume_impact

        return base_slippage

    def calculate_buy_costs(self,
                            trade_value: float,
                            avg_volume: float = 1_000_000,
                            quantity: int = 0) -> Dict[str, float]:
        """Calculate all costs for a buy trade."""
        order_size = quantity if quantity > 0 else 0

        brokerage = self.calculate_brokerage(trade_value)
        stt = self.calculate_stt(trade_value, is_buy=True)
        exchange = self.calculate_exchange_charges(trade_value)
        gst = self.calculate_gst(brokerage, exchange)
        stamp = self.calculate_stamp_duty(trade_value, is_buy=True)
        sebi = self.calculate_sebi_fee(trade_value)
        slippage = self.calculate_slippage(trade_value, avg_volume, order_size)

        total = brokerage + stt + exchange + gst + stamp + sebi + slippage

        return {
            'brokerage': brokerage,
            'stt': stt,
            'exchange_txn': exchange,
            'gst': gst,
            'stamp_duty': stamp,
            'sebi_fee': sebi,
            'slippage': slippage,
            'total': total,
            'total_pct': total / trade_value if trade_value > 0 else 0
        }

    def calculate_sell_costs(self,
                             trade_value: float,
                             avg_volume: float = 1_000_000,
                             quantity: int = 0) -> Dict[str, float]:
        """Calculate all costs for a sell trade."""
        order_size = quantity if quantity > 0 else 0

        brokerage = self.calculate_brokerage(trade_value)
        stt = self.calculate_stt(trade_value, is_buy=False)
        exchange = self.calculate_exchange_charges(trade_value)
        gst = self.calculate_gst(brokerage, exchange)
        stamp = self.calculate_stamp_duty(trade_value, is_buy=False)
        sebi = self.calculate_sebi_fee(trade_value)
        slippage = self.calculate_slippage(trade_value, avg_volume, order_size)

        total = brokerage + stt + exchange + gst + stamp + sebi + slippage

        return {
            'brokerage': brokerage,
            'stt': stt,
            'exchange_txn': exchange,
            'gst': gst,
            'stamp_duty': stamp,
            'sebi_fee': sebi,
            'slippage': slippage,
            'total': total,
            'total_pct': total / trade_value if trade_value > 0 else 0
        }

    def calculate_round_trip(self,
                             entry_price: float,
                             exit_price: float,
                             quantity: int,
                             avg_volume: float = 1_000_000) -> CostBreakdown:
        """
        Calculate complete round-trip costs (buy + sell).

        Args:
            entry_price: Buy price per share
            exit_price: Sell price per share
            quantity: Number of shares
            avg_volume: Average daily volume for slippage estimation

        Returns:
            CostBreakdown with detailed cost analysis
        """
        buy_value = entry_price * quantity
        sell_value = exit_price * quantity

        buy_costs = self.calculate_buy_costs(buy_value, avg_volume, quantity)
        sell_costs = self.calculate_sell_costs(sell_value, avg_volume, quantity)

        total_round_trip = buy_costs['total'] + sell_costs['total']
        avg_value = (buy_value + sell_value) / 2

        return CostBreakdown(
            brokerage=buy_costs['brokerage'] + sell_costs['brokerage'],
            stt=buy_costs['stt'] + sell_costs['stt'],
            exchange_txn=buy_costs['exchange_txn'] + sell_costs['exchange_txn'],
            gst=buy_costs['gst'] + sell_costs['gst'],
            stamp_duty=buy_costs['stamp_duty'] + sell_costs['stamp_duty'],
            sebi_fee=buy_costs['sebi_fee'] + sell_costs['sebi_fee'],
            slippage=buy_costs['slippage'] + sell_costs['slippage'],
            total_buy_cost=buy_costs['total'],
            total_sell_cost=sell_costs['total'],
            total_round_trip=total_round_trip,
            total_pct=total_round_trip / avg_value if avg_value > 0 else 0
        )

    def min_profitable_move(self, trade_value: float = 100000) -> float:
        """
        Calculate minimum price move needed to break even after costs.

        This is the EDGE you need just to cover costs.
        """
        costs = self.calculate_round_trip(
            entry_price=100,
            exit_price=100,
            quantity=int(trade_value / 100)
        )
        return costs.total_pct


class CostAwareBacktester:
    """
    Backtester that properly accounts for transaction costs.

    Many strategies that look profitable in backtests fail in live trading
    because costs weren't properly modeled.
    """

    def __init__(self, costs: Optional[TransactionCosts] = None):
        self.costs = costs or TransactionCosts()
        self.trades: List[Dict] = []

    def simulate_trade(self,
                       entry_price: float,
                       exit_price: float,
                       quantity: int,
                       avg_volume: float = 1_000_000,
                       entry_date: Optional[str] = None,
                       exit_date: Optional[str] = None) -> Dict:
        """
        Simulate a single trade with realistic costs.

        Returns:
            Dictionary with gross/net P&L, costs, and metrics
        """
        # Gross P&L (before costs)
        gross_pnl = (exit_price - entry_price) * quantity
        gross_return = (exit_price / entry_price - 1) if entry_price > 0 else 0

        # Calculate costs
        cost_breakdown = self.costs.calculate_round_trip(
            entry_price, exit_price, quantity, avg_volume
        )

        # Net P&L (after costs)
        net_pnl = gross_pnl - cost_breakdown.total_round_trip
        net_return = net_pnl / (entry_price * quantity) if entry_price > 0 else 0

        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'gross_pnl': gross_pnl,
            'gross_return': gross_return,
            'costs': cost_breakdown.total_round_trip,
            'cost_breakdown': cost_breakdown.to_dict(),
            'net_pnl': net_pnl,
            'net_return': net_return,
            'cost_impact': cost_breakdown.total_pct,
            'profitable_gross': gross_pnl > 0,
            'profitable_net': net_pnl > 0
        }

        self.trades.append(trade)
        return trade

    def simulate_strategy(self,
                          signals: pd.DataFrame,
                          prices: pd.DataFrame,
                          capital: float = 100000,
                          position_size_pct: float = 0.1) -> Dict:
        """
        Simulate a complete trading strategy with costs.

        Args:
            signals: DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
            prices: DataFrame with 'Close' and 'Volume' columns
            capital: Starting capital
            position_size_pct: Percentage of capital per trade

        Returns:
            Strategy performance metrics
        """
        self.trades = []
        current_capital = capital
        position = None
        total_trades = 0
        winning_trades = 0

        for i in range(len(signals)):
            date = signals.index[i]
            signal = signals['signal'].iloc[i]
            price = prices['Close'].iloc[i]
            volume = prices['Volume'].iloc[i] if 'Volume' in prices else 1_000_000

            if signal == 1 and position is None:
                # Open long position
                trade_value = current_capital * position_size_pct
                quantity = int(trade_value / price)

                if quantity > 0:
                    position = {
                        'entry_price': price,
                        'quantity': quantity,
                        'entry_date': str(date.date()) if hasattr(date, 'date') else str(date)
                    }
                    # Deduct buy costs from capital
                    buy_costs = self.costs.calculate_buy_costs(
                        price * quantity, volume, quantity
                    )
                    current_capital -= buy_costs['total']

            elif signal == -1 and position is not None:
                # Close long position
                trade = self.simulate_trade(
                    position['entry_price'],
                    price,
                    position['quantity'],
                    volume,
                    position['entry_date'],
                    str(date.date()) if hasattr(date, 'date') else str(date)
                )

                current_capital += trade['net_pnl']
                total_trades += 1
                if trade['net_pnl'] > 0:
                    winning_trades += 1

                position = None

        # Close any open position
        if position is not None and len(prices) > 0:
            final_price = prices['Close'].iloc[-1]
            final_volume = prices['Volume'].iloc[-1] if 'Volume' in prices else 1_000_000
            trade = self.simulate_trade(
                position['entry_price'],
                final_price,
                position['quantity'],
                final_volume
            )
            current_capital += trade['net_pnl']
            total_trades += 1
            if trade['net_pnl'] > 0:
                winning_trades += 1

        # Calculate metrics
        total_gross_pnl = sum(t['gross_pnl'] for t in self.trades)
        total_net_pnl = sum(t['net_pnl'] for t in self.trades)
        total_costs = sum(t['costs'] for t in self.trades)

        return {
            'starting_capital': capital,
            'ending_capital': current_capital,
            'total_return': (current_capital / capital - 1) if capital > 0 else 0,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'gross_pnl': total_gross_pnl,
            'net_pnl': total_net_pnl,
            'total_costs': total_costs,
            'cost_drag': total_costs / capital if capital > 0 else 0,
            'trades_profitable_gross': sum(1 for t in self.trades if t['profitable_gross']),
            'trades_profitable_net': sum(1 for t in self.trades if t['profitable_net']),
            'gross_to_net_conversion': total_net_pnl / total_gross_pnl if total_gross_pnl != 0 else 0
        }

    def get_cost_analysis(self) -> Dict:
        """Get comprehensive cost analysis from all trades."""
        if not self.trades:
            return {}

        costs_list = [t['costs'] for t in self.trades]
        gross_list = [t['gross_pnl'] for t in self.trades]
        net_list = [t['net_pnl'] for t in self.trades]

        # Count trades that were profitable gross but unprofitable net
        edge_eaten = sum(
            1 for t in self.trades
            if t['profitable_gross'] and not t['profitable_net']
        )

        return {
            'total_trades': len(self.trades),
            'total_costs': sum(costs_list),
            'avg_cost_per_trade': np.mean(costs_list),
            'avg_cost_pct': np.mean([t['cost_impact'] for t in self.trades]),
            'total_gross_pnl': sum(gross_list),
            'total_net_pnl': sum(net_list),
            'cost_as_pct_of_gross': sum(costs_list) / abs(sum(gross_list)) if sum(gross_list) != 0 else 0,
            'trades_with_edge_eaten_by_costs': edge_eaten,
            'edge_eaten_pct': edge_eaten / len(self.trades) if self.trades else 0
        }


def demo():
    """Demonstrate transaction cost calculations."""
    print("=" * 60)
    print("TransactionCosts Demo - Indian Market")
    print("=" * 60)

    costs = TransactionCosts(segment=TradingSegment.EQUITY_DELIVERY)

    # Example trade
    entry_price = 1000
    exit_price = 1050  # 5% profit
    quantity = 100
    trade_value = entry_price * quantity

    print(f"\nExample Trade:")
    print(f"  Entry: Rs {entry_price:,} x {quantity} shares = Rs {trade_value:,}")
    print(f"  Exit: Rs {exit_price:,} x {quantity} shares = Rs {exit_price * quantity:,}")
    print(f"  Gross Profit: Rs {(exit_price - entry_price) * quantity:,}")

    breakdown = costs.calculate_round_trip(entry_price, exit_price, quantity)

    print(f"\nCost Breakdown:")
    print(f"  Brokerage: Rs {breakdown.brokerage:.2f}")
    print(f"  STT: Rs {breakdown.stt:.2f}")
    print(f"  Exchange: Rs {breakdown.exchange_txn:.2f}")
    print(f"  GST: Rs {breakdown.gst:.2f}")
    print(f"  Stamp Duty: Rs {breakdown.stamp_duty:.2f}")
    print(f"  SEBI Fee: Rs {breakdown.sebi_fee:.2f}")
    print(f"  Slippage: Rs {breakdown.slippage:.2f}")
    print(f"  ---")
    print(f"  Total Costs: Rs {breakdown.total_round_trip:.2f}")
    print(f"  Cost %: {breakdown.total_pct:.4%}")

    net_profit = (exit_price - entry_price) * quantity - breakdown.total_round_trip
    print(f"\n  Net Profit: Rs {net_profit:.2f}")

    # Minimum profitable move
    min_move = costs.min_profitable_move(100000)
    print(f"\n  Minimum move to break even: {min_move:.3%}")
    print(f"  (Any edge smaller than this will be eaten by costs)")


class TransactionCostCalculator:
    """
    Simplified wrapper for transaction cost calculations.

    Provides a cleaner interface for the unified predictor.
    """

    def __init__(self, broker_type: BrokerType = BrokerType.DISCOUNT):
        self.costs = TransactionCosts(broker_type=broker_type)

    def calculate_buy_cost(self,
                           price: float,
                           shares: int,
                           is_intraday: bool = False) -> float:
        """
        Calculate total cost of buying shares.

        Returns: Total amount to pay (price * shares + costs)
        """
        if is_intraday:
            self.costs.segment = TradingSegment.EQUITY_INTRADAY
        else:
            self.costs.segment = TradingSegment.EQUITY_DELIVERY

        trade_value = price * shares
        costs = self.costs.calculate_buy_costs(trade_value, quantity=shares)
        return trade_value + costs['total']

    def calculate_sell_proceeds(self,
                                  price: float,
                                  shares: int,
                                  is_intraday: bool = False) -> float:
        """
        Calculate net proceeds from selling shares.

        Returns: Net amount received (price * shares - costs)
        """
        if is_intraday:
            self.costs.segment = TradingSegment.EQUITY_INTRADAY
        else:
            self.costs.segment = TradingSegment.EQUITY_DELIVERY

        trade_value = price * shares
        costs = self.costs.calculate_sell_costs(trade_value, quantity=shares)
        return trade_value - costs['total']

    def get_round_trip_cost_pct(self,
                                 entry_price: float,
                                 exit_price: float,
                                 shares: int = 100) -> float:
        """Get total round-trip cost as percentage."""
        breakdown = self.costs.calculate_round_trip(entry_price, exit_price, shares)
        return breakdown.total_pct

    def adjust_target_for_costs(self,
                                 entry_price: float,
                                 target_price: float,
                                 shares: int = 100) -> float:
        """
        Adjust target price to account for transaction costs.

        Returns the target price that would give the same net profit
        after accounting for costs.
        """
        # Calculate expected gross profit
        gross_profit_pct = (target_price - entry_price) / entry_price

        # Get round-trip costs
        cost_pct = self.get_round_trip_cost_pct(entry_price, target_price, shares)

        # Adjust target to account for costs
        net_profit_pct = gross_profit_pct - cost_pct

        # If target needs to be higher to achieve same net profit
        adjusted_target = entry_price * (1 + gross_profit_pct + cost_pct)

        return round(adjusted_target, 2)


if __name__ == "__main__":
    demo()
