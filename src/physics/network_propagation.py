"""
Network Propagation Model (Wave Mechanics)

Based on wave propagation and interference:
- Market moves spread through correlated networks (sectors)
- Lead-lag relationships exist between stocks
- Constructive interference = aligned signals strengthen prediction
- Destructive interference = conflicting signals weaken prediction

Key Signals:
- Sector leader moving = followers likely to follow
- High coherence = aligned sector moves
- Network momentum = aggregate signal from related stocks
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class NetworkResult:
    """Result from network propagation analysis."""
    is_leader: bool                # Is this stock a sector leader?
    lead_lag_days: float           # Lead (+) or lag (-) vs sector
    propagation_strength: float    # How strongly signals propagate (0-1)
    network_momentum: float        # Aggregate network signal (-1 to 1)
    sector_coherence: float        # Are related stocks aligned? (0-1)
    interference_type: str         # 'constructive', 'destructive', 'neutral'
    expected_follow_through: float # Expected move based on network (%)
    sector: str                    # Identified sector
    correlated_stocks: List[str]   # Most correlated peers
    probability_score: float       # Contribution to final prediction (0-1)
    signal: str                    # 'LEADER_BULLISH', 'FOLLOWING_TREND', 'DIVERGENCE'
    reason: str                    # Human-readable explanation


class NetworkPropagationModel:
    """
    Implements network/wave propagation for stock prediction.

    The model:
    1. Identifies stock's sector and correlated peers
    2. Detects lead-lag relationships
    3. Calculates network momentum (aggregate signal)
    4. Measures sector coherence (alignment)
    """

    # Sector definitions (NSE stocks)
    SECTOR_STOCKS = {
        'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'PERSISTENT', 'COFORGE', 'MPHASIS'],
        'BANKING': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK', 'INDUSINDBK', 'BANDHANBNK', 'FEDERALBNK', 'IDFCFIRSTB'],
        'NBFC': ['BAJFINANCE', 'BAJAJFINSV', 'SBICARD', 'CHOLAFIN', 'SHRIRAMFIN', 'M&MFIN', 'MUTHOOTFIN'],
        'AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO', 'ASHOKLEY', 'TVSMOTOR'],
        'PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP', 'BIOCON', 'LUPIN', 'AUROPHARMA'],
        'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO', 'COLPAL', 'GODREJCP'],
        'METAL': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'COALINDIA', 'NMDC', 'SAIL'],
        'ENERGY': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'POWERGRID', 'NTPC', 'ADANIGREEN'],
        'REALTY': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE', 'SOBHA'],
        'INFRA': ['LT', 'ADANIPORTS', 'ULTRACEMCO', 'GRASIM', 'ACC', 'AMBUJACEM', 'SHREECEM'],
    }

    # Known sector leaders (typically highest market cap, most liquid)
    SECTOR_LEADERS = {
        'IT': ['TCS', 'INFY'],
        'BANKING': ['HDFCBANK', 'ICICIBANK'],
        'NBFC': ['BAJFINANCE'],
        'AUTO': ['MARUTI', 'TATAMOTORS'],
        'PHARMA': ['SUNPHARMA', 'DRREDDY'],
        'FMCG': ['HINDUNILVR', 'ITC'],
        'METAL': ['TATASTEEL', 'JSWSTEEL'],
        'ENERGY': ['RELIANCE'],
        'REALTY': ['DLF'],
        'INFRA': ['LT'],
    }

    def __init__(
        self,
        correlation_threshold: float = 0.5,
        momentum_period: int = 5,
        lead_lag_max: int = 3
    ):
        self.correlation_threshold = correlation_threshold
        self.momentum_period = momentum_period
        self.lead_lag_max = lead_lag_max

    def identify_sector(self, symbol: str) -> Optional[str]:
        """Identify which sector a stock belongs to."""
        symbol_upper = symbol.upper().replace('.NS', '').replace('.BO', '')

        for sector, stocks in self.SECTOR_STOCKS.items():
            if symbol_upper in stocks:
                return sector

        return None

    def is_sector_leader(self, symbol: str, sector: str) -> bool:
        """Check if stock is a sector leader."""
        symbol_upper = symbol.upper().replace('.NS', '').replace('.BO', '')
        leaders = self.SECTOR_LEADERS.get(sector, [])
        return symbol_upper in leaders

    def get_sector_peers(self, symbol: str, sector: str) -> List[str]:
        """Get peer stocks in same sector."""
        symbol_upper = symbol.upper().replace('.NS', '').replace('.BO', '')
        peers = self.SECTOR_STOCKS.get(sector, [])
        return [p for p in peers if p != symbol_upper]

    def calculate_network_momentum(
        self,
        symbol: str,
        sector_returns: Dict[str, pd.Series]
    ) -> float:
        """
        Calculate weighted momentum from network peers.

        Network momentum = correlation-weighted sum of peer returns
        """
        symbol_upper = symbol.upper().replace('.NS', '').replace('.BO', '')

        if symbol_upper not in sector_returns:
            return 0.0

        target_returns = sector_returns[symbol_upper]
        network_signal = 0.0
        total_weight = 0.0

        for other_symbol, other_returns in sector_returns.items():
            if other_symbol == symbol_upper:
                continue

            # Align series
            aligned = pd.concat([target_returns, other_returns], axis=1).dropna()
            if len(aligned) < 20:
                continue

            # Calculate correlation
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

            if pd.isna(corr) or abs(corr) < self.correlation_threshold:
                continue

            # Recent return of related stock
            recent_return = other_returns.iloc[-self.momentum_period:].sum()

            # Weight by absolute correlation
            network_signal += abs(corr) * recent_return
            total_weight += abs(corr)

        if total_weight > 0:
            return network_signal / total_weight

        return 0.0

    def calculate_sector_coherence(
        self,
        sector_returns: Dict[str, pd.Series]
    ) -> float:
        """
        Calculate how aligned sector stocks are.

        Coherence = fraction of stocks moving in same direction
        """
        if len(sector_returns) < 2:
            return 0.5

        recent_directions = []

        for returns in sector_returns.values():
            if len(returns) >= self.momentum_period:
                recent_return = returns.iloc[-self.momentum_period:].sum()
                direction = 1 if recent_return > 0 else -1
                recent_directions.append(direction)

        if not recent_directions:
            return 0.5

        # Coherence = what fraction agree with majority
        majority = 1 if sum(recent_directions) > 0 else -1
        coherence = sum(1 for d in recent_directions if d == majority) / len(recent_directions)

        return coherence

    def determine_interference(self, coherence: float) -> str:
        """Classify interference pattern."""
        if coherence >= 0.75:
            return 'constructive'
        elif coherence <= 0.4:
            return 'destructive'
        else:
            return 'neutral'

    def calculate_propagation_strength(
        self,
        symbol: str,
        df: pd.DataFrame,
        sector_data: Dict[str, pd.DataFrame]
    ) -> float:
        """
        Calculate how strongly price moves propagate.

        Based on volume and correlation.
        """
        symbol_upper = symbol.upper().replace('.NS', '').replace('.BO', '')

        if symbol_upper not in sector_data or df is None:
            return 0.5

        # Volume factor
        own_volume = df['volume'].mean()

        all_volumes = []
        for sym, data in sector_data.items():
            if 'volume' in data.columns:
                all_volumes.append(data['volume'].mean())

        if not all_volumes:
            return 0.5

        avg_volume = np.mean(all_volumes)
        volume_ratio = own_volume / (avg_volume + 1e-6)
        volume_factor = min(np.sqrt(volume_ratio), 2.0)

        # Correlation factor (already calculated via coherence)
        # Stronger stocks have more influence

        propagation = volume_factor * 0.5  # Base propagation
        return np.clip(propagation, 0.1, 1.0)

    def get_signal_label(
        self,
        is_leader: bool,
        coherence: float,
        network_momentum: float,
        interference: str
    ) -> Tuple[str, str]:
        """Determine signal label and reason."""

        if is_leader and coherence > 0.7 and network_momentum > 0.01:
            return 'LEADER_BULLISH', 'Sector leader with aligned bullish sector'
        elif is_leader and coherence > 0.7 and network_momentum < -0.01:
            return 'LEADER_BEARISH', 'Sector leader with aligned bearish sector'
        elif not is_leader and interference == 'constructive' and network_momentum > 0.01:
            return 'FOLLOWING_BULLISH', 'Following sector bullish momentum'
        elif not is_leader and interference == 'constructive' and network_momentum < -0.01:
            return 'FOLLOWING_BEARISH', 'Following sector bearish momentum'
        elif interference == 'destructive':
            return 'DIVERGENCE', 'Sector signals conflicting - low conviction'
        else:
            return 'NEUTRAL_NETWORK', 'No clear network signal'

    def score(
        self,
        symbol: str,
        df: pd.DataFrame,
        sector_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> NetworkResult:
        """
        Calculate comprehensive network propagation score.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV for target stock
            sector_data: Dict of DataFrames for sector peers

        Returns:
            NetworkResult with network analysis
        """
        symbol_upper = symbol.upper().replace('.NS', '').replace('.BO', '')

        # Identify sector
        sector = self.identify_sector(symbol)

        if sector is None:
            return self._default_result(symbol_upper, 'Stock sector not identified')

        # Check if leader
        is_leader = self.is_sector_leader(symbol, sector)
        lead_lag = 0 if is_leader else 0.5  # Leaders have 0 lag

        # Get peers
        peers = self.get_sector_peers(symbol, sector)

        # If no sector data provided, use limited analysis
        if sector_data is None or len(sector_data) < 2:
            return self._limited_result(symbol_upper, sector, is_leader, peers)

        # Calculate returns for all stocks
        sector_returns = {}
        for sym, data in sector_data.items():
            if 'close' in data.columns and len(data) >= 20:
                returns = data['close'].pct_change()
                sector_returns[sym.upper().replace('.NS', '').replace('.BO', '')] = returns

        # Add target stock returns
        if len(df) >= 20:
            sector_returns[symbol_upper] = df['close'].pct_change()

        if len(sector_returns) < 2:
            return self._limited_result(symbol_upper, sector, is_leader, peers)

        # Calculate network metrics
        network_momentum = self.calculate_network_momentum(symbol_upper, sector_returns)
        coherence = self.calculate_sector_coherence(sector_returns)
        interference = self.determine_interference(coherence)
        propagation = self.calculate_propagation_strength(symbol, df, sector_data)

        # Expected follow-through
        if interference == 'constructive':
            expected_follow = network_momentum * 0.5  # 50% of network signal
        elif interference == 'destructive':
            expected_follow = network_momentum * 0.1  # Only 10%
        else:
            expected_follow = network_momentum * 0.3

        # Signal and reason
        signal, reason = self.get_signal_label(is_leader, coherence, network_momentum, interference)

        # Probability score
        # Leaders in coherent bullish sectors = high probability
        # Constructive interference + bullish momentum = higher probability
        if is_leader and coherence > 0.7:
            base_prob = 0.60
        elif coherence > 0.7:
            base_prob = 0.58
        elif interference == 'destructive':
            base_prob = 0.48
        else:
            base_prob = 0.52

        # Adjust for momentum direction
        if network_momentum > 0.02:
            prob_score = base_prob + min(network_momentum * 2, 0.1)
        elif network_momentum < -0.02:
            prob_score = base_prob - min(abs(network_momentum) * 2, 0.1)
        else:
            prob_score = base_prob

        prob_score = np.clip(prob_score, 0.35, 0.70)

        # Find most correlated peers
        correlated = peers[:3] if len(peers) >= 3 else peers

        return NetworkResult(
            is_leader=is_leader,
            lead_lag_days=lead_lag,
            propagation_strength=propagation,
            network_momentum=network_momentum,
            sector_coherence=coherence,
            interference_type=interference,
            expected_follow_through=expected_follow * 100,
            sector=sector,
            correlated_stocks=correlated,
            probability_score=prob_score,
            signal=signal,
            reason=reason
        )

    def _limited_result(
        self,
        symbol: str,
        sector: str,
        is_leader: bool,
        peers: List[str]
    ) -> NetworkResult:
        """Return limited result when full sector data not available."""
        return NetworkResult(
            is_leader=is_leader,
            lead_lag_days=0 if is_leader else 0.5,
            propagation_strength=0.5,
            network_momentum=0,
            sector_coherence=0.5,
            interference_type='neutral',
            expected_follow_through=0,
            sector=sector,
            correlated_stocks=peers[:3] if peers else [],
            probability_score=0.52 if is_leader else 0.50,
            signal='LIMITED_DATA',
            reason=f'Sector {sector} identified, but peer data not available'
        )

    def _default_result(self, symbol: str, reason: str) -> NetworkResult:
        """Return neutral result when analysis isn't possible."""
        return NetworkResult(
            is_leader=False,
            lead_lag_days=0,
            propagation_strength=0.5,
            network_momentum=0,
            sector_coherence=0.5,
            interference_type='neutral',
            expected_follow_through=0,
            sector='UNKNOWN',
            correlated_stocks=[],
            probability_score=0.50,
            signal='NO_NETWORK',
            reason=reason
        )
