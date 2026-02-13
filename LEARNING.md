# LEARNING.md - Algo Trading Reference Guide

This document consolidates learnings from analyzing successful open-source algo trading repositories and our own stock-predict codebase audit.

---

## Table of Contents
1. [Code Audit Results](#code-audit-results)
2. [Reference Repos Overview](#reference-repos-overview)
3. [Key Strategies Explained](#key-strategies-explained)
4. [Technical Indicators - Correct Implementations](#technical-indicators)
5. [Risk Management Patterns](#risk-management-patterns)
6. [Broker Integration Patterns](#broker-integration-patterns)
7. [Action Items for Our System](#action-items)

---

## 1. Code Audit Results {#code-audit-results}

### Overall Score: 7/10 - Good Foundation, Needs Improvements

### What's Correct

| Component | Status | Notes |
|-----------|--------|-------|
| RSI Formula | ✅ | Standard Wilder's RSI using EMA |
| MACD Formula | ✅ | Correct 12/26/9 EMA implementation |
| Bollinger Bands | ✅ | 20 SMA ± 2 std dev |
| ATR Formula | ✅ | Proper True Range calculation |
| Position Sizing | ✅ | Risk-based with 10% cap |
| Backtest Engine | ✅ | No lookahead bias |

### Issues Found

| Issue | File | Impact | Priority |
|-------|------|--------|----------|
| SELL signals disabled | `scoring.py:162-170` | Only LONG trades | Medium |
| 57.6% win rate | Backtest results | Marginal edge | HIGH |
| No Supertrend | `technical.py` | Missing key indicator | HIGH |
| Same logic for Intraday/Swing | `intraday_model.py` | No intraday-specific signals | HIGH |
| Volume not confirming price | `scoring.py` | Weak breakout validation | Medium |
| ADX not enforced | `scoring.py` | No trend filter | Medium |

### Key Insight from Backtest
```
BUY Signal Accuracy: 57.6% (330 trades)
SELL Signal Accuracy: 28.6% (DISABLED - inverted)

57.6% is marginal. Need 60%+ after broker fees to be profitable.
```

---

## 2. Reference Repos Overview {#reference-repos-overview}

### Downloaded Repos (in `reference_repos/`)

| Repo | Broker | Strategy | Stars | Key Feature |
|------|--------|----------|-------|-------------|
| `nifty_bot` | Upstox | 9:30 AM Breakout | Active | Telegram control, Crash recovery |
| `vwap_supertrend_bot` | Upstox | VWAP+MACD+Supertrend | Active | Triple confirmation |
| `alice_blue_futures` | Alice Blue | Supertrend+RSI | Active | Multi-timeframe |
| `algo_strategies_india` | Zerodha | Short Straddle | Active | Options selling |
| `zerodha_ai_ml` | Zerodha | RSI+Guppy | 500+ | AI/ML enhanced |

---

## 3. Key Strategies Explained {#key-strategies-explained}

### Strategy 1: 9:30 AM Breakout (from `nifty_bot`)

**Logic:**
```
1. 9:25-9:30: Observation Phase
   - Scan option chain for strikes near target premium (₹180)
   - Select CE and PE closest to target

2. 9:30-9:35: Entry Window (STRICT - only 5 minutes)
   - Wait for price to break above trigger price
   - SUSTAIN CHECK: Wait 5 seconds, verify price still above trigger
   - If sustained → ENTER

3. Exit Rules:
   - Target: Entry + 40 points
   - Stop Loss: Entry - 20 points
   - Time Exit: 10:00 AM (hard stop)
   - Trailing: After +20 points, move SL to entry (risk-free)
```

**Key Code Pattern:**
```python
# From nifty_bot/core/strategy.py
if ltp > trigger_price:
    time.sleep(5)  # SUSTAIN LOGIC - wick trap protection
    verified_ltp = broker.get_ltp(instrument_key)
    if verified_ltp > trigger_price:
        self._execute_trade(...)  # Confirmed breakout
```

**Why It Works:**
- Limited entry window prevents overtrading
- Sustain check filters false breakouts (wicks)
- Hard time exit avoids theta decay
- Risk-free move protects profits

---

### Strategy 2: VWAP + MACD + Supertrend Triple Confirmation (from `vwap_supertrend_bot`)

**Logic:**
```
Entry Conditions (ALL must be true):
1. Supertrend changes direction (flip signal)
2. MACD crossover (bullish/bearish)
3. Price near VWAP (within 0.2%)

If all 3 conditions align → ENTER
```

**Key Code Pattern:**
```python
# From brahmastra.py
st_change = last['supertrend'] != prev['supertrend']
macd_cross = (prev['macd'] < prev['signal'] and last['macd'] > last['signal'])
price_near_vwap = abs(last['close'] - last['vwap']) < (last['vwap'] * 0.002)

if st_change and macd_cross and price_near_vwap:
    # Triple confirmation - HIGH probability trade
    execute_trade(...)
```

**Why It Works:**
- Multiple confirmations reduce false signals
- VWAP acts as institutional fair value
- Supertrend provides trend direction
- MACD confirms momentum

---

### Strategy 3: Supertrend + RSI (from `alice_blue_futures`)

**Parameters:**
```python
supertrend_period = 7
supertrend_multiplier = 2.5
rsi_period = 7
```

**Logic:**
```
BUY Signal:
- Supertrend direction = "up" (price above ST line)
- RSI not overbought (< 70)

SELL Signal:
- Supertrend direction = "down" (price below ST line)
- RSI not oversold (> 30)
```

**Supertrend Calculation (Correct Implementation):**
```python
# From ab_lib.py - CORRECT Supertrend Algorithm
def SuperTrend(df, period=7, multiplier=2.5):
    # 1. Calculate ATR
    ATR(df, period)

    # 2. Calculate bands
    df['basic_ub'] = (high + low) / 2 + multiplier * atr
    df['basic_lb'] = (high + low) / 2 - multiplier * atr

    # 3. Final bands (the key logic)
    for i in range(period, len(df)):
        # Upper band only moves DOWN (resistance)
        df['final_ub'][i] = min(df['basic_ub'][i], df['final_ub'][i-1])
                           if close[i-1] > df['final_ub'][i-1]
                           else df['basic_ub'][i]

        # Lower band only moves UP (support)
        df['final_lb'][i] = max(df['basic_lb'][i], df['final_lb'][i-1])
                           if close[i-1] < df['final_lb'][i-1]
                           else df['basic_lb'][i]

    # 4. Determine trend
    df['STX'] = 'up' if close > ST else 'down'
```

---

### Strategy 4: Short Straddle (from `algo_strategies_india`)

**Logic:**
```
At 9:59 AM:
1. Get Bank Nifty LTP
2. Calculate ATM strike (round to nearest 100)
3. SELL ATM CE + SELL ATM PE (Short Straddle)
4. Place stop loss orders:
   - CE SL: Entry + 50 points
   - PE SL: Entry + 60 points

At 12:30 PM (Re-entry check):
- If both legs hit SL → Re-enter fresh straddle

At 3:06 PM:
- Square off all positions
```

**Key Code Pattern:**
```python
# From bank_nifty_fixed_stop_loss_short_straddle.py
def get_banknifty_atm_strike(ltp):
    r = ltp % 100
    if r < 50:
        atm = ltp - r
    else:
        atm = ltp - r + 100
    return int(atm)

# Entry
ce_order_id = marketorder_sell(ce_symbol, lots * 25)
pe_order_id = marketorder_sell(pe_symbol, lots * 25)

# Stop Loss
ce_sl_orderid = stoploss_order_buy(ce_symbol, lots*25, ce_sell_price + 50)
pe_sl_orderid = stoploss_order_buy(pe_symbol, lots*25, pe_sell_price + 60)
```

---

## 4. Technical Indicators - Correct Implementations {#technical-indicators}

### Supertrend (MISSING from our codebase - MUST ADD)

```python
def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    Supertrend Indicator - Most reliable intraday indicator.

    Returns: Series with Supertrend values and direction
    """
    # 1. Calculate ATR
    hl2 = (df['high'] + df['low']) / 2
    prev_close = df['close'].shift(1)

    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()

    # 2. Calculate bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    # 3. Initialize
    supertrend = pd.Series([np.nan] * len(df), index=df.index)
    direction = pd.Series([1] * len(df), index=df.index)  # 1=up, -1=down

    # 4. Calculate Supertrend
    for i in range(1, len(df)):
        # Trend flip logic
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1  # Uptrend
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1  # Downtrend
        else:
            direction.iloc[i] = direction.iloc[i-1]
            # Band stickiness
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]

    # 5. Set Supertrend line
    for i in range(len(df)):
        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend, direction
```

### VWAP (Correct Implementation)

```python
def calculate_vwap(df):
    """
    Volume Weighted Average Price.
    Resets daily for intraday trading.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_pv = (typical_price * df['volume']).cumsum()
    cumulative_volume = df['volume'].cumsum()
    vwap = cumulative_pv / cumulative_volume
    return vwap
```

### ADX (Trend Strength Filter)

```python
def calculate_adx(df, period=14):
    """
    Average Directional Index - Measures trend strength.

    ADX > 25: Strong trend (follow it)
    ADX < 25: Weak trend (range-bound, mean reversion)
    """
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    atr = calculate_atr(df, period)

    plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period).mean()

    return adx
```

---

## 5. Risk Management Patterns {#risk-management-patterns}

### Pattern 1: Weekly Loss Limit (Kill Switch)

```python
# From nifty_bot/config.py
WEEKLY_MAX_LOSS = 10000.0  # Stop trading if weekly PnL < -₹10,000

# In strategy:
weekly_pnl = get_weekly_pnl()
if weekly_pnl < -abs(WEEKLY_MAX_LOSS):
    self.entry_locked = True  # KILL SWITCH
    alert("Weekly Max Loss Hit. Trading Disabled until Monday.")
```

### Pattern 2: Hard Stop Loss (Our Requirements)

```python
# Our requirements: ₹1L capital, stop at ₹80K
RISK_CONFIG = {
    "initial_capital": 100000,
    "hard_stop_loss": 80000,      # GLOBAL KILL SWITCH
    "max_loss_per_trade": 2000,   # 2% per trade
    "max_positions": 3,
    "position_size_pct": 0.25,    # 25% per position
}

def can_trade(self):
    """Check if we should continue trading"""
    current_value = self.get_portfolio_value()
    if current_value < self.hard_stop_loss:
        self.shutdown("Portfolio below ₹80K. STOPPED.")
        return False
    return True
```

### Pattern 3: Trailing Stop Loss

```python
# From nifty_bot - Risk-free move + candle trailing
def manage_trailing(self, ltp, entry):
    # Phase 1: Risk-free move at +20 points
    if not self.risk_free_done and ltp >= entry + 20:
        self.update_sl(entry)  # Move SL to entry
        self.risk_free_done = True

    # Phase 2: Candle-based trailing after 9:45
    if current_time >= time(9, 45):
        candles = broker.get_candles('5minute', 2)
        candle_low = candles[-1]['low']
        if candle_low > current_sl:
            self.update_sl(candle_low)  # Trail to candle low
```

### Pattern 4: Position Sizing (Kelly Criterion)

```python
# From robust_predictor.py
def calculate_position_size(prob_win, win_rate, loss_rate, capital):
    # Kelly: f* = (bp - q) / b
    b = win_rate / loss_rate
    kelly = (b * prob_win - (1 - prob_win)) / b

    # Half-Kelly for safety
    kelly = max(0, min(0.25, kelly * 0.5))

    return capital * kelly
```

---

## 6. Broker Integration Patterns {#broker-integration-patterns}

### Upstox API Pattern (from `nifty_bot`)

```python
# infra/upstox_client.py
class UpstoxBroker:
    def __init__(self, access_token):
        self.base_url = 'https://api.upstox.com/v2'
        self.headers = {'Authorization': f'Bearer {access_token}'}

    def get_ltp(self, instrument_key):
        url = f"{self.base_url}/market-quote/ltp"
        params = {'instrument_key': instrument_key}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json()['data'][instrument_key]['last_price']

    def place_order(self, instrument_key, side, quantity, order_type, **kwargs):
        url = f"{self.base_url}/order/place"
        payload = {
            'instrument_token': instrument_key,
            'transaction_type': side,
            'quantity': quantity,
            'order_type': order_type,
            'product': 'I',  # Intraday
            **kwargs
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json()['data']['order_id']
```

### Zerodha Kite API Pattern (from `algo_strategies_india`)

```python
from kiteconnect import KiteConnect

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

def place_market_order(symbol, side, quantity):
    return kite.place_order(
        tradingsymbol=symbol,
        exchange=kite.EXCHANGE_NFO,
        transaction_type=side,
        quantity=quantity,
        order_type=kite.ORDER_TYPE_MARKET,
        product=kite.PRODUCT_MIS,  # Intraday
        variety=kite.VARIETY_REGULAR
    )

def place_sl_order(symbol, quantity, trigger_price):
    return kite.place_order(
        tradingsymbol=symbol,
        exchange=kite.EXCHANGE_NFO,
        transaction_type=kite.TRANSACTION_TYPE_BUY,
        quantity=quantity,
        order_type=kite.ORDER_TYPE_SLM,  # Stop Loss Market
        product=kite.PRODUCT_MIS,
        variety=kite.VARIETY_REGULAR,
        trigger_price=trigger_price
    )
```

---

## 7. Action Items for Our System {#action-items}

### HIGH Priority (Before Live Trading)

- [ ] **Add Supertrend Indicator** to `technical.py`
- [ ] **Add ADX Trend Filter** - Only trade when ADX > 25
- [ ] **Improve BUY accuracy to 60%+** through better filters
- [ ] **Add intraday-specific logic**:
  - Opening range breakout (9:15-9:30)
  - VWAP cross strategy
  - First 15-min high/low

### Medium Priority

- [ ] **Volume confirmation** - Require volume > 1.5x average for breakouts
- [ ] **Add Broker module** - Zerodha or Upstox integration
- [ ] **Add Risk Manager** with hard stop at ₹80K
- [ ] **Paper trading mode** before live

### Nice to Have

- [ ] **Telegram alerts** for trade notifications
- [ ] **F&O support** - Options trading capability
- [ ] **Multiple strategy support** - Breakout, Straddle, etc.

---

## Quick Reference: Configuration Values from Successful Bots

| Parameter | nifty_bot | alice_blue | algo_strategies | Recommended |
|-----------|-----------|------------|-----------------|-------------|
| Supertrend Period | - | 7 | - | 7-10 |
| Supertrend Multiplier | - | 2.5 | - | 2.5-3.0 |
| RSI Period | - | 7 | - | 7-14 |
| Entry Window | 9:30-9:35 | Market hours | 9:59 | 9:30-9:45 |
| Target Points | 40 | - | - | 1.5x SL |
| Stop Loss Points | 20 | - | 50-60 | 1-2% |
| Max Loss/Week | ₹10,000 | - | - | ₹20,000 (20%) |
| Position Size | 1 lot | - | 5 lots | 25% capital |

---

*Last Updated: February 5, 2026*
*Generated from analysis of 5 open-source trading repos*
