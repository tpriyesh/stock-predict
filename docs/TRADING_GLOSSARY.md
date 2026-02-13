# Trading Concepts & How Our System Uses Them

A complete guide to every trading concept in this codebase - what it means in real trading, and exactly how our algo uses it.

---

## Table of Contents

1. [Core Trading Concepts](#1-core-trading-concepts)
2. [Technical Indicators](#2-technical-indicators)
3. [Order Types & Execution](#3-order-types--execution)
4. [Risk Management](#4-risk-management)
5. [Trading Phases & Timing](#5-trading-phases--timing)
6. [Broker & Account Concepts](#6-broker--account-concepts)
7. [System Architecture Concepts](#7-system-architecture-concepts)
8. [Our Trading Flow (Step by Step)](#8-our-trading-flow-step-by-step)

---

## 1. Core Trading Concepts

### LTP (Last Traded Price)

**What it means:** The most recent price at which a stock was bought/sold on the exchange. This is the "current price" you see on any stock app.

**In our code:** `broker.get_ltp("RELIANCE")` returns a single float. Used everywhere - checking stop losses, calculating P&L, deciding entries.

**Files:** `broker/base.py:155`, `agent/orchestrator.py:569`

---

### OHLCV (Open, High, Low, Close, Volume)

**What it means:**
- **Open** - First traded price of the day
- **High** - Highest price during the day
- **Low** - Lowest price during the day
- **Close** - Last traded price of the day
- **Volume** - Total number of shares traded

These 5 values summarize one day's (or one candle's) trading activity. Every chart you see is made from OHLCV data.

**In our code:** All technical indicators are calculated from OHLCV DataFrames. The `src/features/technical.py` file takes a DataFrame with these 5 columns and adds 30+ indicator columns on top.

**Files:** `src/features/technical.py:42-45`, `broker/base.py:95-106`

---

### Bid and Ask (Spread)

**What it means:**
- **Bid** - Highest price someone is willing to BUY at right now
- **Ask** - Lowest price someone is willing to SELL at right now
- **Spread** = Ask - Bid (the gap between them)

If RELIANCE bid is 2500 and ask is 2501, you'd buy at 2501 and sell at 2500. The spread is the "cost" of trading.

**In our code:** Returned in `Quote` object from `broker.get_quote()`. We use MARKET orders which fill at the ask (for buys), so we accept the spread cost.

**Files:** `broker/base.py:104-105`, `broker/zerodha.py:243-244`

---

### Long Position (Going Long)

**What it means:** Buying a stock expecting the price to go UP. You buy at 100, price goes to 110, you sell for Rs.10 profit per share.

**In our code:** All our trades are long (BUY only). `ENABLE_SELL_SIGNALS = False` in config. We buy stocks, hold them during the day, and sell before market close.

**Files:** `config/trading_config.py:140-141`

---

### Short Position (Going Short)

**What it means:** Selling a stock you don't own, expecting the price to go DOWN. You sell at 100, price drops to 90, you buy back for Rs.10 profit. Very risky because losses are unlimited (price can go up forever).

**In our code:** DISABLED. `enable_sell_signals: False`. The trailing stop calculator supports shorts (`is_long=False`) but we never use it. Too risky for our capital size.

---

### Intraday Trading

**What it means:** Buying and selling within the SAME trading day. You never hold overnight. All positions must be closed (squared off) before 3:30 PM.

**Why:** No overnight risk (news, global events while market is closed). Lower margin required. But also means you can't ride multi-day trends.

**In our code:** Everything is intraday. We use `ProductType.INTRADAY` (MIS on Zerodha, I on Upstox). Forced square-off at 3:00 PM. Entry window is 9:30 AM - 2:30 PM only.

**Files:** `config/trading_config.py:103-117`, `agent/orchestrator.py:240-260`

---

### Square Off

**What it means:** Closing all open positions. For intraday, this means selling everything you bought. If you don't do it by 3:15-3:20 PM, the broker does it automatically (with penalty charges).

**In our code:** At 3:00 PM (`SQUARE_OFF_TIME`), the orchestrator enters `SQUARE_OFF` phase and sells all positions with market orders. We do it 30 min before market close to avoid the broker's auto-square-off.

**Files:** `agent/orchestrator.py:250-260`, `config/trading_config.py:112-113`

---

### P&L (Profit & Loss)

**What it means:**
- **Realized P&L** - Profit/loss from trades you've completed (bought and sold)
- **Unrealized P&L** - Paper profit/loss on positions you still hold (not yet sold)
- **Total P&L** = Realized + Unrealized

**In our code:** `PnL` dataclass has all three. Realized is tracked internally by the orchestrator (sum of all closed trades). Unrealized comes from broker (current price vs entry price of open positions). Kill switch uses total P&L.

**Files:** `broker/base.py:117-122`, `risk/manager.py:85-104`

---

### Slippage

**What it means:** The difference between the price you expected and the price you actually got. If you place a MARKET buy when LTP is 100 but you get filled at 100.50, that's 0.5% slippage. Happens because price moves between your decision and execution.

**In our code:** Config has `MAX_SLIPPAGE_PCT = 0.005` (0.5%). We use MARKET orders for speed (guaranteed fill, but slippage possible). Fill price comes from `order_book.average_price` after execution.

**Files:** `config/trading_config.py:207-208`, `agent/orchestrator.py:414-422`

---

## 2. Technical Indicators

These are mathematical calculations on OHLCV data that help predict future price movement. Our system calculates ALL of these, then feeds them to the signal generator.

### RSI (Relative Strength Index)

**What it means:** Measures how fast price is rising vs falling. Scale 0-100.
- **RSI > 70** = "Overbought" - price may be too high, could fall
- **RSI < 30** = "Oversold" - price may be too low, could rise
- **RSI 30-70** = Normal range

**Formula:** `RSI = 100 - (100 / (1 + RS))` where `RS = Avg Gain / Avg Loss` over 14 days.

**In our code:** `TechnicalIndicators.rsi(prices, period=14)`. Used in signal generation to detect oversold stocks (buying opportunity). RSI < 30 + upward momentum = potential BUY signal.

**Real trader usage:** "RELIANCE RSI hit 25, it's oversold. I'll buy expecting a bounce."

**Files:** `src/features/technical.py:133-156`, `config/trading_config.py:168-176`

---

### MACD (Moving Average Convergence Divergence)

**What it means:** Shows the relationship between two moving averages of price. Detects trend changes.

- **MACD Line** = 12-day EMA minus 26-day EMA
- **Signal Line** = 9-day EMA of the MACD line
- **Histogram** = MACD Line minus Signal Line

When MACD crosses above Signal = bullish (price may rise).
When MACD crosses below Signal = bearish (price may fall).

**In our code:** `TechnicalIndicators.macd()`. The histogram tells us momentum direction and strength. Positive and growing histogram = bullish momentum.

**Real trader usage:** "MACD just crossed above signal line on INFY - trend is turning bullish."

**Files:** `src/features/technical.py:158-183`

---

### Bollinger Bands

**What it means:** A price channel that widens when volatile, narrows when calm.

- **Middle Band** = 20-day SMA (average price)
- **Upper Band** = Middle + 2 standard deviations
- **Lower Band** = Middle - 2 standard deviations

Price stays within bands ~95% of the time. Touching lower band = potential buying opportunity.

**Our extra metrics:**
- **BB Width** = How wide the bands are (high = volatile, low = calm)
- **BB Position** = Where price is within the bands (0 = at lower, 1 = at upper, 0.5 = middle)

**In our code:** `TechnicalIndicators.bollinger_bands()`. BB Position < 0.2 means price is near the lower band (oversold zone). Used in conjunction with RSI for stronger signals.

**Real trader usage:** "TCS touched the lower Bollinger Band with RSI at 28 - strong buy signal."

**Files:** `src/features/technical.py:185-208`, lines 70-73 for BB width/position

---

### ATR (Average True Range)

**What it means:** Measures how much a stock moves per day on average (volatility). NOT direction, just magnitude.

**Formula:** Average of True Range over 14 days, where True Range = max of:
1. Today's High - Today's Low
2. |Today's High - Yesterday's Close|
3. |Today's Low - Yesterday's Close|

If RELIANCE ATR is 30, it means the stock typically moves Rs.30 per day.

**In our code:** This is CRITICAL. ATR is used for:
1. **Setting stop losses** - Stop loss distance based on ATR (e.g., 1.5x ATR below entry)
2. **Trailing stops** - ATR-based trailing adjusts with volatility
3. **Position sizing** - Higher ATR = smaller position (more volatile = less shares)
4. **Trend strength** - ATR used in ADX calculation

**Real trader usage:** "SBIN ATR is 15. I'll place my stop loss 22 points below entry (1.5x ATR)."

**Files:** `src/features/technical.py:210-232`, `agent/orchestrator.py:490-510`

---

### SMA & EMA (Moving Averages)

**What it means:**
- **SMA (Simple Moving Average)** - Average of last N closing prices. SMA 20 = avg of last 20 days.
- **EMA (Exponential Moving Average)** - Like SMA but gives more weight to recent prices. Reacts faster to changes.

Used to identify trends:
- Price above SMA 20 > SMA 50 = Uptrend
- Price below SMA 20 < SMA 50 = Downtrend

**In our code:** We calculate SMA 20, 50, 200 and EMA 20. Also calculate how far price is from each MA (percentage). Used in trend detection and signal generation.

- `price_vs_sma20` = how far price is from 20-day average (positive = above, bullish)
- `price_vs_sma200` = long-term trend indicator

**Real trader usage:** "HDFCBANK is trading 3% above its 50-day SMA with rising volume - strong uptrend."

**Files:** `src/features/technical.py:79-91`

---

### VWAP (Volume Weighted Average Price)

**What it means:** Average price weighted by volume. If most trading happened at 100, VWAP is near 100 even if price briefly touched 110.

**Why it matters:** Institutional traders use VWAP as a benchmark. Price above VWAP = buyers in control. Below = sellers.

**In our code:** Calculated with daily reset (resets each day for intraday relevance). Used as a reference level for signal generation.

**Files:** `src/features/technical.py:98-107`

---

### Volume Ratio

**What it means:** Today's volume divided by the 20-day average volume.
- Volume Ratio > 1.5 = Unusually high volume (something is happening)
- Volume Ratio < 0.5 = Very low volume (no interest)

High volume + price movement = strong signal. Low volume + price movement = unreliable.

**In our code:** `volume_ratio = volume / volume_sma_20`. Division-by-zero protected. High volume ratio makes buy signals more confident.

**Real trader usage:** "TITAN volume is 3x normal with price rising - institutional buying confirmed."

**Files:** `src/features/technical.py:93-96`, `config/trading_config.py:186`

---

### Momentum

**What it means:** How much price has changed over a period. Positive momentum = price rising. Negative = falling.

- **Momentum 5** = Price change over 5 days (%)
- **Momentum 10** = Price change over 10 days (%)
- **Momentum 20** = Price change over 20 days (%)

**In our code:** `pct_change(N) * 100`. Used to measure strength and direction of recent price moves.

**Files:** `src/features/technical.py:109-112`

---

### Support & Resistance

**What it means:**
- **Support** - Price level where the stock tends to stop falling (buyers step in). Floor.
- **Resistance** - Price level where the stock tends to stop rising (sellers step in). Ceiling.

**In our code:** Simplified as 20-day rolling min (support) and max (resistance). Used to identify key price levels for entry/exit decisions.

**Real trader usage:** "RELIANCE has support at 2450. If it bounces from there, I'll buy."

**Files:** `src/features/technical.py:234-247`

---

### Trend Strength (ADX-like)

**What it means:** ADX (Average Directional Index) measures HOW STRONG a trend is, not its direction. Scale 0-100.
- **< 25** = Weak trend / ranging market (don't trade trends)
- **25-50** = Moderate trend (okay to trade)
- **> 50** = Strong trend (great for trend-following)

**In our code:** Simplified ADX calculation using directional movement. Used to filter out ranging markets where trend-following strategies fail.

**Files:** `src/features/technical.py:249-280`

---

### Candlestick Patterns

**What they mean:** Visual patterns formed by 1-3 candles that hint at future direction:

| Pattern | Shape | Signal | What it means |
|---------|-------|--------|---------------|
| **Doji** | Tiny body (open = close) | Indecision | Trend may reverse |
| **Hammer** | Long lower shadow, small body at top | Bullish | Sellers tried to push down but buyers pushed back |
| **Shooting Star** | Long upper shadow, small body at bottom | Bearish | Buyers tried to push up but sellers won |
| **Bullish Engulfing** | Big green candle swallows previous red | Bullish | Strong buying pressure |
| **Bearish Engulfing** | Big red candle swallows previous green | Bearish | Strong selling pressure |
| **Morning Star** | 3-candle: big red, tiny, big green | Bullish | Bottom reversal pattern |

**In our code:** Each pattern detected as 1 (present) or 0 (absent). Fed into the signal generator as additional confidence factors. A buy signal with a hammer pattern gets higher confidence.

**Files:** `src/features/technical.py:282-348`

---

### Supertrend

**What it means:** A trend-following indicator that plots a line above or below price:
- When price is ABOVE the Supertrend line = Uptrend (BUY zone)
- When price CROSSES BELOW = Sell signal
- When price CROSSES ABOVE = Buy signal

Uses ATR to set the distance. More volatile stocks get wider bands.

**In our code:** Configurable period (7) and multiplier (2.5). Used as one of the trend confirmation signals.

**Files:** `config/trading_config.py:161-166`

---

## 3. Order Types & Execution

### Market Order

**What it means:** Buy/sell immediately at whatever the current price is. You're guaranteed a fill but not a specific price. Best for when speed matters.

**In our code:** All entries and exits use `OrderType.MARKET`. We prioritize guaranteed fills over price control because intraday positions MUST be closed before 3:30 PM.

**Files:** `broker/base.py:20`, `agent/orchestrator.py:384-386`

---

### Limit Order

**What it means:** Buy/sell only at a specific price or better. "Buy RELIANCE at 2500 or lower." May never fill if price doesn't reach your limit.

**In our code:** Supported (`OrderType.LIMIT`) but we use MARKET orders for entries/exits. Limit orders are too slow for our intraday strategy.

---

### Stop Loss (SL) Order

**What it means:** An order that activates only when price reaches a "trigger price." Used to limit losses.

**Example:** You buy RELIANCE at 2500. You place a Stop Loss order with trigger at 2450. If price drops to 2450, the SL order activates and sells your shares automatically, capping your loss at Rs.50/share.

**Two types in our system:**
- **SL** (`OrderType.SL`) - Stop Loss Limit: When triggered, places a LIMIT order (may not fill in a crash)
- **SL-M** (`OrderType.SL_M`) - Stop Loss Market: When triggered, places a MARKET order (guaranteed fill, used by us)

**In our code:** After every BUY entry, we immediately place an SL-M order on the broker. This protects us even if our system crashes - the broker executes the SL automatically.

**Files:** `broker/base.py:22-23`, `agent/orchestrator.py:512-541`

---

### Product Type (MIS vs CNC)

**What it means:**
- **MIS (Margin Intraday Squareoff)** / **INTRADAY** - Must be squared off same day. Lower margin required (you can trade with less capital). Auto-squared-off by broker at 3:15-3:20 PM.
- **CNC (Cash and Carry)** / **DELIVERY** - Hold for multiple days. Full margin required. No auto-square-off.

**In our code:** Always `ProductType.INTRADAY` (maps to "MIS" on Zerodha, "I" on Upstox). We never hold overnight.

**Files:** `broker/base.py:35-38`, `broker/zerodha.py:54-57`

---

### Order Book

**What it means:** The complete list of all orders you've placed today with their current status (filled, pending, cancelled, etc.).

**In our code:** `broker.get_order_book()` returns all today's orders. `broker.get_order_status(order_id)` returns a specific order's details including fill price, filled quantity, etc.

**Files:** `broker/base.py:63-78`

---

### Fill / Fill Price

**What it means:** When your order gets executed, it's "filled." The fill price is the actual price you got. For MARKET orders, fill price may differ slightly from LTP (slippage).

**In our code:** After placing an order, we call `get_order_status()` and read `order_book.average_price` as the fill price. This becomes the entry price for our TradeRecord.

**Files:** `agent/orchestrator.py:414-422`

---

### Partial Fill

**What it means:** You wanted to buy 100 shares but only 60 were available at market. You got 60 filled, 40 still pending. Common with less liquid stocks.

**In our code:** `OrderStatus.PARTIAL_FILL` is handled. If a partial fill happens, we use `order_book.filled_quantity` as the actual position size.

**Files:** `broker/base.py:30`, `agent/orchestrator.py:398-422`

---

## 4. Risk Management

### Kill Switch

**What it means:** Emergency brake. If total portfolio value drops below a hard floor, ALL trading stops immediately. Prevents catastrophic losses.

**In our code:** If portfolio value < Rs.80,000 (we started with Rs.1,00,000), the kill switch activates:
1. Blocks all new entries
2. Exits all open positions
3. Sends CRITICAL Telegram alert
4. Sets `is_killed = True` - bot stops

**Real trader equivalent:** "If my account drops to 80K, I'm done for the month."

**Files:** `risk/manager.py:106-140`, `config/trading_config.py:67-68`

---

### Daily Loss Limit

**What it means:** Maximum loss allowed in a single day. If daily P&L drops below this, stop trading for the rest of the day but resume tomorrow.

**In our code:** `MAX_DAILY_LOSS_PCT = 5%` of initial capital = Rs.5,000. If we lose Rs.5,000 in a day, no more entries. Existing positions are still managed (stops/exits).

**Why:** Prevents "revenge trading" - after a bad day, you tend to take bigger risks to recover. Daily limit protects against this.

**Files:** `risk/manager.py:54`, `config/trading_config.py:70-71`

---

### Per-Trade Risk

**What it means:** Maximum you can lose on any single trade, as a percentage of capital.

**In our code:** `MAX_PER_TRADE_RISK_PCT = 2%` of capital = Rs.2,000. If a trade has stop loss Rs.50 below entry, maximum shares = Rs.2000 / Rs.50 = 40 shares. This determines position size.

**Real trader rule:** "Never risk more than 2% of your account on one trade."

**Files:** `risk/manager.py:55`, `config/trading_config.py:73-74`

---

### Position Sizing

**What it means:** How many shares to buy. NOT random - calculated based on risk.

**Our formula:**
```
Risk amount = Capital * max_per_trade_risk_pct  (e.g., 1,00,000 * 2% = 2,000)
Risk per share = Entry price - Stop loss          (e.g., 500 - 490 = 10)
Max quantity = Risk amount / Risk per share       (e.g., 2000 / 10 = 200 shares)
```

Also capped by:
- **Concentration limit:** Max 25% of capital in one stock
- **Max positions:** 3 simultaneous positions
- **Hard cap:** 10,000 shares max per trade
- **Minimum:** signal_qty vs risk_qty, whichever is smaller

**Files:** `risk/manager.py:155-210`, `agent/orchestrator.py:371-378`

---

### Risk-Reward Ratio (R:R)

**What it means:** How much you can make vs how much you can lose.

```
Risk = Entry - Stop Loss = potential loss per share
Reward = Target - Entry = potential gain per share
R:R Ratio = Reward / Risk
```

**Example:** Buy at 100, Stop at 95, Target at 115.
- Risk = 5, Reward = 15
- R:R = 15/5 = 3:1 (you make 3x what you risk)

**In our code:** Minimum R:R = 1.5:1. We reject any signal where the potential reward is less than 1.5x the risk. This ensures that even with 50% accuracy, we're profitable long-term.

**Files:** `config/trading_config.py:133-134`, `agent/signal_adapter.py` (validation)

---

### Trailing Stop Loss

**What it means:** A stop loss that MOVES UP as price rises, locking in profits. It never moves down.

**Example:** Buy at 100, initial stop at 95. Price rises to 110 - trailing stop moves to 107. Price rises to 115 - stop moves to 112. Price falls to 112 - stop triggers, you sell at 112 (locked in Rs.12 profit instead of Rs.0 if stop stayed at 95).

**Our methods:**
| Method | How it works | When used |
|--------|-------------|-----------|
| **Percentage** | Stop = Highest Price - X% | Default (0.5% trail) |
| **ATR Multiple** | Stop = Highest - 1.5x ATR | Adapts to volatility |
| **Chandelier** | Stop = Highest - 3x ATR | Wider, for trending stocks |
| **Step** | Moves in fixed R increments | Locks profit in steps |

**Activation:** Trailing only starts after price moves 1R in profit (configurable). Before that, original stop loss stays.

**Ratchet behavior:** Stop ONLY moves up, never down. `max(current_stop, new_stop)` ensures this.

**Files:** `src/execution/trailing_stop.py` (full file), `agent/orchestrator.py:601-624`

---

### R (Risk Unit)

**What it means:** A unit of measurement based on your initial risk.

```
1R = Entry Price - Stop Loss = what you risked
```

**Example:** Buy at 100, stop at 95. Your 1R = 5.
- If price goes to 107.50 = you're at 1.5R profit (made 1.5x what you risked)
- If price goes to 95 = you lost 1R (your initial risk)

**In our code:** Trailing stop activation is based on R. Default: trailing starts after 1R profit. Step trailing moves stop in 0.5R increments.

**Files:** `src/execution/trailing_stop.py:80-92`

---

### Confidence Score

**What it means:** How confident our algo is in a signal. Scale 0-100%.

Built from multiple factors:
- Technical indicators (RSI, MACD, Bollinger, etc.)
- News sentiment
- Volume confirmation
- Trend alignment

**In our code:** Minimum confidence = 60%. Any signal below 60% is rejected. Higher confidence = more reliable signal (historically).

**Files:** `config/trading_config.py:130-131`, `agent/signal_adapter.py`

---

### Reconciliation

**What it means:** Comparing what our system THINKS we have vs what the broker ACTUALLY has. Catches mismatches from network errors, manual trades, or broker-side corrections.

**In our code:** Runs every 2 minutes:
1. Gets positions from broker
2. Compares with `active_trades` dict
3. If broker has position we don't know about = "orphaned" (alert)
4. If we track a position broker doesn't have = "externally closed" (clean up)

**Files:** `agent/orchestrator.py:555-566`

---

## 5. Trading Phases & Timing

Our system operates in distinct phases throughout the day:

| Phase | Time | What happens |
|-------|------|-------------|
| **PRE_MARKET** | 8:45 - 9:15 | System boots, checks token, waits for market |
| **MARKET_OPEN** | 9:15 - 9:30 | Market opens but we DON'T trade yet (first 15 min is volatile noise) |
| **ENTRY_WINDOW** | 9:30 - 14:30 | We look for signals and enter positions |
| **HOLDING** | When we have positions | Monitor stops, trail, check news |
| **SQUARE_OFF** | 14:30 - 15:00 | No new entries. Start closing positions |
| **MARKET_CLOSED** | After 15:30 | Everything closed, generate daily report |

**Why skip first 15 minutes?** Market open is chaotic - gap ups/downs, pending orders from overnight. Prices stabilize after 9:30.

**Why stop entries at 14:30?** Need enough time for the trade to work before forced square-off.

**Files:** `agent/orchestrator.py:45-51`, `config/trading_config.py:94-120`

---

## 6. Broker & Account Concepts

### Margin

**What it means:** Money "locked" by the broker when you take a position.

- **Available Cash** - Money you can use for new trades
- **Used Margin** - Money locked in current positions
- **Total Balance** = Available + Used Margin

For intraday (MIS), brokers give leverage - you might need only 20% margin. So with Rs.1 lakh, you could control Rs.5 lakh worth of stocks. But our system doesn't use excessive leverage.

**In our code:** `Funds` dataclass has `available_cash`, `used_margin`, `total_balance`. Position sizing uses `available_cash` to decide how many shares we can buy.

**Files:** `broker/base.py:109-114`

---

### Access Token (API Authentication)

**What it means:** A temporary password that lets our code talk to the broker's API. Expires daily (both Zerodha and Upstox).

**Zerodha flow:** Every morning, our script automatically:
1. Sends username + password to Zerodha
2. Generates a TOTP code (like Google Authenticator)
3. Gets a temporary token back
4. Uses that token for all API calls that day

**In our code:** `scripts/zerodha_auto_login.py` handles this. Can run as cron job at 7:30 AM. Token saved to `~/.zerodha_token.json` and `.env`.

**Files:** `scripts/zerodha_auto_login.py`, `broker/zerodha.py:67-82`

---

### TOTP (Time-Based One-Time Password)

**What it means:** A 6-digit code that changes every 30 seconds. Same as Google Authenticator. Zerodha uses this for 2FA (two-factor authentication).

**In our code:** We have the TOTP secret key saved in `.env`. The `pyotp` library generates the current code: `pyotp.TOTP(secret_key).now()`. This lets us login without opening a browser or phone.

**Files:** `scripts/zerodha_auto_login.py:148`

---

### Rate Limiting

**What it means:** Brokers limit how many API calls you can make per second. Zerodha allows 10 requests/second. Exceeding this gets you temporarily blocked.

**In our code:** `BrokerRateLimiter` tracks calls per endpoint. Before every API call, `limiter.wait()` sleeps if needed. Prevents us from getting blocked.

**Files:** `utils/rate_limiter.py`, `broker/zerodha.py` (8 RPS conservative limit)

---

### Circuit Breaker

**What it means (in our code):** If an API endpoint fails 5 times in a row, we stop calling it for 60 seconds. Prevents hammering a broken API and getting banned.

**Not to be confused with:** Exchange circuit breakers (NSE halts trading if Nifty moves >10%) - that's different.

**In our code:** `CircuitBreakerRegistry` tracks per-endpoint failures. After `failure_threshold` consecutive failures, the circuit "opens" and rejects all calls for `recovery_timeout` seconds.

**Files:** `utils/error_handler.py`, `broker/zerodha.py:88-92`

---

### Paper Trading

**What it means:** Simulated trading with fake money. Real market data, fake orders. Used to test strategies before risking real capital.

**In our code:** `PaperBroker` fetches real prices from yfinance but simulates order execution locally. Starts with Rs.1 lakh virtual capital. SL orders trigger based on real prices.

**Files:** `broker/paper.py`

---

## 7. System Architecture Concepts

### Signal Generation Pipeline

```
Price Data (yfinance)
    |
    v
Technical Indicators (30+ indicators calculated)
    |
    v
News Sentiment (NewsAPI, GNews, RSS feeds)
    |
    v
Signal Generator (combines everything, scores confidence)
    |
    v
Signal Adapter (validates: min confidence, R:R ratio, stop distance)
    |
    v
Orchestrator (executes: place order, set stop loss, track position)
```

---

### Trade Lifecycle

```
1. SIGNAL: Generator says "BUY RELIANCE at 2500, SL 2475, Target 2550"

2. VALIDATION:
   - Risk manager: "2% risk check OK, max 80 shares"
   - Signal adapter: "Confidence 68% > 60% minimum, R:R 2:1 > 1.5, OK"

3. ENTRY:
   - Place MARKET BUY order for 80 shares
   - Wait for fill (2 seconds max)
   - Record fill price (say 2501)
   - Place SL-M order at 2475

4. MONITORING (every 30-60 seconds):
   - Check LTP
   - If LTP < stop (2475) -> EXIT
   - If LTP >= target (2550) -> EXIT
   - If LTP rising -> trail stop up
   - Check news for held stock

5. EXIT:
   - Place MARKET SELL order
   - Cancel the SL order
   - Record P&L
   - Save to database

6. REPORTING:
   - Update daily P&L
   - Send Telegram notification
   - Persist to SQLite
```

---

### Database Persistence

**Why:** If the system crashes mid-trade, we need to know what positions are open.

**What's stored:**
- Every trade entry (symbol, price, quantity, time)
- Every trade exit (exit price, P&L, reason)
- Stop loss updates
- Daily summaries
- Position snapshots

**In our code:** SQLite database at `data/trades.db`. Written to immediately on every trade event. On restart, reconciliation compares DB with broker to catch any gaps.

**Files:** `utils/trade_db.py`

---

### Telegram Alerts

**What it means:** Instant notifications to your phone for critical events.

**Events that trigger alerts:**
| Event | Priority | Message |
|-------|----------|---------|
| Trade entry | INFO | "BUY RELIANCE 80 @ 2501" |
| Trade exit | INFO | "SELL RELIANCE +Rs.3,920 (Target hit)" |
| Kill switch | CRITICAL | "Portfolio below 80K - ALL TRADING STOPPED" |
| Token expiry | CRITICAL | "Token expired - emergency exit + shutdown" |
| Exit order failed | CRITICAL | "Position stuck open - manual intervention!" |
| Reconciliation mismatch | WARNING | "Broker has INFY position we don't track" |

**Files:** `utils/alerts.py`

---

## 8. Our Trading Flow (Step by Step)

Here's exactly what happens when you run `python start.py --zerodha`:

```
7:30 AM  - Cron runs zerodha_auto_login.py (refreshes token via TOTP)
8:45 AM  - start.py begins
           1. Acquires file lock (prevents double instances)
           2. Checks if today is a trading day (not weekend/holiday)
           3. Validates Zerodha token (auto-refreshes if expired)
           4. Waits for market to open

9:15 AM  - Market opens, orchestrator starts
           - Phase: MARKET_OPEN (watching but NOT trading)

9:30 AM  - Phase: ENTRY_WINDOW (now accepting signals)
           - Every 5 minutes: fetch new signals for 20 watchlist stocks
           - Each signal checked: confidence > 60%? R:R > 1.5? Risk OK?
           - If valid signal found: BUY + place SL order

9:30-2:30 - Phase: ENTRY_WINDOW + HOLDING
           - Every 30 seconds: check prices of held positions
           - Trail stops up on winning trades
           - Exit on stop loss or target hit
           - Every 5 minutes: check news for held stocks
           - Every 2 minutes: reconcile with broker

2:30 PM  - Phase: SQUARE_OFF
           - No new entries allowed
           - Start exiting remaining positions

3:00 PM  - Force exit ALL positions (MARKET sell orders)

3:30 PM  - Market closes
           - Generate daily report
           - Send Telegram summary
           - Save daily summary to DB
           - Release file lock, exit

Done. See you tomorrow!
```

---

## Quick Reference: Config Values

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Initial Capital | Rs.1,00,000 | Starting money |
| Hard Stop | Rs.80,000 | Kill switch level (20% drawdown) |
| Max Daily Loss | 5% (Rs.5,000) | Stop trading for the day |
| Per-Trade Risk | 2% (Rs.2,000) | Max loss on single trade |
| Max Position | 25% (Rs.25,000) | Max in one stock |
| Max Positions | 3 | Max simultaneous trades |
| Min Confidence | 60% | Reject weak signals |
| Min R:R | 1.5:1 | Reject bad risk/reward |
| Entry Window | 9:30 - 14:30 | When we can buy |
| Square Off | 15:00 | When we must sell |
| Trailing Start | After 1R profit | When trailing activates |
| Trail Distance | 0.5% | How far stop trails below price |
| Watchlist | 20 NSE stocks | RELIANCE, TCS, HDFC, INFY, etc. |
