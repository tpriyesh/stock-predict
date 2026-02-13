# Trading Agent Architecture

## Overview

Automated intraday trading agent for NSE stocks. Runs daily on your laptop, executes trades via Upstox or Zerodha API, and stops after market close.

- **Capital**: Configurable (default Rs.1,00,000)
- **Target**: 2-5% daily return
- **Hard Stop**: Kills all trading if portfolio drops below threshold
- **Broker**: Upstox, Zerodha (live) / Paper (testing)
- **Signal Engine**: 7-model ensemble (Base + Physics + Math + HMM + Macro + Alternative Data + Advanced Math)

---

## Quick Start

```bash
# One-time: Get broker credentials
python scripts/upstox_auth.py      # For Upstox
python scripts/zerodha_auto_login.py # For Zerodha

# Daily: Run this every morning
python start.py              # paper mode (safe)
python start.py --live       # real money (Upstox)
python start.py --live --broker zerodha  # real money (Zerodha)
```

---

## How It Works (Daily Flow)

```
YOU run "python start.py" in the morning
         |
         v
+-------------------------------------------------------------+
|  1. CHECK TRADING DAY                                        |
|     - Weekend? Holiday? -> Exit                              |
|     - Broker token valid? -> If expired, tells you to fix    |
+-------------------------------------------------------------+
|  2. WAIT FOR MARKET (if started early)                       |
|     - Sleeps until 8:45 AM (near-zero CPU)                   |
+-------------------------------------------------------------+
|  3. PRE-MARKET (8:45 - 9:15)                                 |
|     - Runs stock-predict signal generator (7-model ensemble) |
|     - Fetches prices for 20 NIFTY 50 stocks                  |
|     - Calculates technical indicators (RSI, MACD, BB, ATR)   |
|     - Fetches and analyzes news sentiment                    |
|     - Ranks stocks by confidence score                       |
+-------------------------------------------------------------+
|  4. MARKET OPEN (9:15 - 9:30)                                |
|     - Observes market, does NOT trade yet                    |
|     - Checks any overnight positions                         |
|     - Reconciles broker positions with local state           |
+-------------------------------------------------------------+
|  5. ENTRY WINDOW (9:30 - 14:30)                              |
|     - Places BUY orders for top signals                      |
|     - Max 3 positions at once                                |
|     - Each trade has stop-loss placed immediately            |
|     - Refreshes signals every 5 minutes                      |
|     - Signal filter pipeline:                                |
|       Confidence >= 60% -> R:R >= 1.5 -> Entry window        |
|       -> Price valid -> Supertrend UP -> ADX >= 25 -> PASS   |
+-------------------------------------------------------------+
|  6. HOLDING / ENTRY_WINDOW (monitors positions)              |
|     NOTE: With default config (entry_window_end == 14:30 ==  |
|     square_off_start), HOLDING phase is unreachable. Position|
|     monitoring runs during ENTRY_WINDOW instead.             |
|     - Checks positions every 60 seconds                      |
|     - Stop loss monitored -> EXIT if hit                     |
|     - Target price monitored -> EXIT if hit                  |
|     - Trailing stop updates highest price tracking           |
|     - Trailing stop moves SL up as price rises               |
|     - Broker SL order modified when trailing stop moves      |
|     - News checked every 5 min for held positions            |
|       - Negative news (sentiment < -0.3, 2+ articles)        |
|         -> tighten stop to breakeven                         |
|       - Strongly negative (sentiment < -0.5, 0-1 articles)   |
|         -> immediate exit                                    |
|       Note: if/elif means -0.5 check only fires when         |
|       there are fewer than 2 articles                        |
|     - All trades persisted to SQLite database                |
|     - Telegram alerts sent on entry/exit                     |
+-------------------------------------------------------------+
|  7. SQUARE OFF (14:30 - 15:00)                               |
|     - Exits ALL remaining positions                          |
|     - No exceptions - everything closed before 3 PM          |
+-------------------------------------------------------------+
|  8. REPORT & EXIT                                            |
|     - Prints daily P&L report                                |
|     - Saves daily summary to database                        |
|     - Sends Telegram daily report                            |
|     - Logs everything to logs/trading_YYYY-MM-DD.log         |
|     - Exits cleanly                                          |
+-------------------------------------------------------------+
```

Press Ctrl+C at any time -> graceful shutdown (exits all positions first).

---

## System Architecture

```
+------------------------------------------------------------------+
|                        .env (Configuration)                       |
|  Capital, risk limits, market hours, thresholds, API keys         |
+-------------------------------+----------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|              config/trading_config.py (CONFIG singleton)          |
|  Reads ALL trading parameters from .env with sensible defaults    |
+-------------------------------+----------------------------------+
                                |
              +-----------------+------------------+
              v                 v                  v
         start.py          trade.py           daemon.py
        (morning)         (CLI tools)        (background)
              |                 |                  |
              +-----------------+------------------+
                                v
+------------------------------------------------------------------+
|                 agent/orchestrator.py                              |
|            TradingOrchestrator (main loop)                         |
|                                                                   |
|  Phases: PRE_MARKET -> MARKET_OPEN -> ENTRY_WINDOW -> SQUARE_OFF  |
|  Note: HOLDING phase exists but is unreachable with defaults      |
|  (entry_window_end == square_off_start == 14:30)                  |
|  Position monitoring + news checks happen during ENTRY_WINDOW too |
|                                                                   |
|  Features:                                                        |
|  - Trade persistence (SQLite) with crash recovery                 |
|  - Position reconciliation (broker vs local state)                |
|  - Stop loss monitoring (checks every position every cycle)       |
|  - Trailing stop (percentage-based, moves SL up as price rises)   |
|  - News monitoring (checks news every 5 min during HOLDING)       |
|  - SL order failure protection (exit if SL can't be placed)       |
|  - Price validation (skips check if broker returns price=0)       |
|  - Telegram alerts for trades, errors, daily reports              |
+--------+-----------------+--------------------+------------------+
         |                 |                    |
         v                 v                    v
+----------------+ +----------------+ +-----------------------------+
| risk/          | | agent/         | | broker/                     |
| manager.py     | | signal_        | | +----------+ +-----------+  |
|                | | adapter.py     | | | paper.py | | upstox.py |  |
| Kill switch    | |                | | | (testing)| | (real $)  |  |
| Daily limits   | | Converts       | | +----------+ +-----------+  |
| Position size  | | signals ->     | |        ^                    |
| Trade valid.   | | trades         | | +--------------------+      |
| PnL tracking   | |                | | | zerodha.py         |      |
+----------------+ | Filters:       | | | (Zerodha Kite)     |      |
                   | - Supertrend   | | +--------------------+      |
                   | - ADX          | |                             |
                   | - Confidence   | | utils/rate_limiter.py       |
                   | - R:R ratio    | | utils/error_handler.py      |
                   | - Entry window | +-----------------------------+
                   +-------+--------+
                           |
                           v
+------------------------------------------------------------------+
|                src/signals/enhanced_generator.py                   |
|              (7-Model Ensemble Signal Engine)                      |
|                                                                   |
|  Pipeline:                                                        |
|  1. Fetch prices (yfinance) for NIFTY 50 universe                 |
|  2. Calculate technicals (RSI, MACD, Bollinger, ATR, VWAP)        |
|  3. Fetch + analyze news (NewsAPI, GNews, RSS)                    |
|  4. Score each stock with 7-model ensemble:                       |
|     - Base Technical (25%)                                        |
|     - Physics Engine (18%)                                        |
|     - Math Engine (14%)                                           |
|     - HMM Regime Detection (13%)                                  |
|     - Macro Engine (10%) - commodities, currencies, bonds         |
|     - Alternative Data (10%) - earnings, options, institutional   |
|     - Advanced Math (10%) - Kalman, Wavelet, PCA, Markov          |
|  5. Rank by confidence, require multi-model agreement             |
|  6. Filter by threshold (>0.60), apply regime filters             |
|  7. Generate explanations for top picks                           |
|  8. Output predictions with entry, SL, target prices              |
+------------------------------------------------------------------+
         |                           |
         v                           v
+-------------------+     +-------------------+
| utils/trade_db.py |     | utils/alerts.py   |
| SQLite persistence|     | Telegram alerts   |
| Crash recovery    |     | Trade notifications|
| Audit trail       |     | Error alerts      |
+-------------------+     +-------------------+
```

---

## Signal Filter Pipeline

Signals pass through a strict multi-gate filter before execution:

```
EnhancedSignalGenerator.run() -> raw predictions (7-model ensemble)
         |
         v
    _convert_prediction() -> TradeSignal
         |
         v
    _passes_filters():
         |
    1. Decision must be BUY or SELL
    2. Confidence >= 0.60 (CONFIG.signals.min_confidence)
    3. Risk:Reward >= 1.5 (CONFIG.signals.min_risk_reward)
    4. Within entry window (9:30 - 14:30)
    5. Entry/SL/Target prices valid (> 0)
    6. SL below entry, target above entry (for BUY)
    7. Supertrend direction == UP for BUY, DOWN for SELL
    8. ADX >= 25 (strong trend required)
         |
         v
    PASSED -> execute trade
```

**Supertrend** (period=7, multiplier=2.5): Best intraday trend indicator. Rejects counter-trend entries which are the primary loss source.

**ADX** (threshold=25): Rejects signals in weak/ranging markets where intraday momentum trading loses money.

Both filters **reject on error** (conservative approach - better to miss a trade than take a bad one).

---

## 7-Model Ensemble Scoring

The enhanced signal engine uses a 7-model ensemble for robust predictions:

| Model | Description | Weight | File |
|-------|-------------|--------|------|
| **Base Technical** | RSI, MACD, Bollinger, Momentum, Volume, News | 25% | `src/models/scoring.py` |
| **Physics Engine** | Momentum conservation, spring reversion, energy clustering | 18% | `src/physics/physics_engine.py` |
| **Math Engine** | Fourier cycles, Hurst exponent, entropy, statistical mechanics | 14% | `src/math_models/math_engine.py` |
| **HMM Regime** | Hidden Markov Model detecting market regimes | 13% | `src/ml/hmm_regime_detector.py` |
| **Macro Engine** | Commodities, currencies, bonds, semiconductor index | 10% | `src/macro/macro_engine.py` |
| **Alternative Data** | Earnings events, options flow, institutional flow | 10% | `src/alternative_data/alternative_engine.py` |
| **Advanced Math** | Kalman filter, Wavelet analysis, PCA, Markov chains, DQN | 10% | `src/core/advanced_engine.py` |

**Multi-Model Agreement:**
- **Strong Signal**: 5-7 models agree (HIGH conviction)
- **Moderate Signal**: 3-4 models agree (MODERATE conviction)
- **Weak Signal**: 1-2 models agree (LOW conviction - filtered out)

**SELL signals disabled** - Historical accuracy was 28.6% (inverted). System generates BUY or HOLD only.

---

## Position Monitoring

Every 60 seconds during ENTRY_WINDOW and HOLDING phases:

```
For each active position:
  1. Get current price from broker
  2. Validate price (skip if 0 or invalid)
  3. Check stop loss hit -> EXIT immediately
  4. Check target hit -> EXIT
  5. Update highest_price tracking
  6. Calculate trailing stop (percentage-based)
  7. If trailing stop > current stop:
     - Update local stop level
     - Modify broker SL order to new level
     - Log the trailing stop movement
     - Alert on SL modification failure

Every 5 minutes during ENTRY_WINDOW/HOLDING:
  For each held position:
    - Fetch fresh news articles (last 4 hours)
    - Analyze sentiment via OpenAI
    - If sentiment < -0.3 AND 2+ articles:
        Tighten stop to breakeven
    - Elif sentiment < -0.5 (0-1 articles only, due to if/elif):
        EXIT immediately
    Note: The if/elif structure means the -0.5 check only
    triggers when there are fewer than 2 articles, since
    2+ articles with sentiment < -0.5 are caught by the
    -0.3 branch first.
```

---

## Module Reference

### Entry Points (3 files, ~1,400 lines)

| File | Lines | Purpose | When to Use |
|------|-------|---------|-------------|
| `start.py` | 477 | Daily morning script | **Every trading day** |
| `trade.py` | 499 | CLI commands | Status checks, testing |
| `daemon.py` | 421 | Background daemon | Alternative to start.py |

**start.py** - The main entry point:
- Checks trading day, validates broker token, waits for market
- Initializes broker -> risk manager -> orchestrator
- Runs until market close, then exits
- Supports `--broker zerodha` flag

**trade.py** - CLI commands:
```bash
python trade.py start [--live] [--broker {upstox,zerodha}]  # Start trading
python trade.py signals [-s SYMBOL]                         # View signals
python trade.py status                                      # Portfolio status
python trade.py positions                                   # Open positions
python trade.py pnl                                         # Today's P&L
python trade.py test [--live] [--broker {upstox,zerodha}]   # Test connection
python trade.py supertrend                                  # Test Supertrend
```

**daemon.py** - Background daemon (alternative):
```bash
python daemon.py run      # Foreground (testing)
python daemon.py start    # Background
python daemon.py stop     # Stop
python daemon.py status   # Check if running
```

---

### Core Trading (3 files, ~2,300 lines)

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `agent/orchestrator.py` | 1,377 | `TradingOrchestrator` | Main trading loop |
| `agent/signal_adapter.py` | 439 | `SignalAdapter` | Predictions -> trade decisions |
| `risk/manager.py` | 501 | `RiskManager` | Capital protection |

**TradingOrchestrator** - The brain:
- Manages 6 trading phases with time-based transitions
- **Trade persistence**: All trades saved to SQLite via `utils/trade_db.py`
- **Position reconciliation**: Syncs broker positions with local state at startup
- **Crash recovery**: Recovers orphaned trades from previous sessions
- Enters positions from SignalAdapter signals
- Places stop-loss orders immediately after entry
- **If SL order fails, exits position immediately** (no unprotected trades)
- Monitors stop loss hit every cycle
- **Trailing stop**: Tracks highest price per position, calculates new stop using `TrailingStopCalculator`, modifies broker SL order when stop moves up
- **News monitoring**: During HOLDING phase, checks news every 5 min for held positions. Negative sentiment tightens stop to breakeven. Strongly negative triggers immediate exit
- **Telegram alerts**: Sends notifications on trades, errors, kill switch, daily reports
- Force-exits everything at square-off time
- Graceful shutdown on Ctrl+C or errors
- **Null-safe**: Handles broker returning None for order status or 0 for price
- **PnL tracking**: Calls `risk_manager.on_trade_complete(pnl)` on every exit

**SignalAdapter** - The bridge:
- Calls `src/signals/enhanced_generator.py` to get predictions (7-model ensemble)
- Converts raw predictions to `TradeSignal` objects
- **Supertrend gate**: BUY signals require Supertrend direction == UP (1), SELL requires DOWN (-1)
- **ADX filter**: Rejects signals when ADX < 25 (weak/ranging market)
- Applies filters: min confidence, min R:R, entry window, price validation
- Calculates recommended position size
- **Price data cache with expiry**: Linked to `signal_cache_minutes`, prevents stale Supertrend data
- Caches signals (configurable TTL)
- **BUY signals only** (SELL disabled due to 28.6% accuracy)
- **Conservative error handling**: Trend filter errors reject the signal (don't allow through)

**RiskManager** - The guardian:
- **Kill switch**: Portfolio < hard stop -> ALL trading stops
- **Daily loss limit**: Exceeded -> no more trades today
- **Per-trade risk**: Max 2% of capital at risk per trade
- **Position sizing**: Based on stop-loss distance and risk budget
- **Stop-loss validation**: Min 0.5%, max 3% distance
- **Concentration limit**: Max 3 concurrent positions
- **Daily PnL tracking**: `on_trade_complete(pnl)` called by orchestrator

---

### Broker Layer (4 files, ~2,150 lines)

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `broker/base.py` | 323 | `BaseBroker` | Abstract interface |
| `broker/paper.py` | 505 | `PaperBroker` | Simulated trading |
| `broker/upstox.py` | 717 | `UpstoxBroker` | Real Upstox API |
| `broker/zerodha.py` | 610 | `ZerodhaBroker` | Real Zerodha Kite Connect API |

**BaseBroker** defines the interface (all brokers implement these):
- `connect()`, `disconnect()`, `is_connected()`
- `get_ltp(symbol)`, `get_quote(symbol)`, `get_quotes(symbols)`
- `place_order(order)`, `modify_order()`, `cancel_order()`
- `get_positions()`, `get_funds()`, `get_pnl()`
- `is_market_open()`, `get_holidays()`

**Data types**: Order, OrderResponse, OrderBook, Position, Quote, Funds, PnL, ProductType

**PaperBroker** - For testing without real money:
- Uses yfinance for real market prices
- Simulates order fills with 0.1% slippage
- Tracks positions and P&L in memory
- `cancel_order()` handles both OPEN and PENDING status
- No API keys needed

**UpstoxBroker** - For real trading with Upstox:
- Upstox v2 REST API integration
- Rate limiting: 200 requests/min, 15 requests/sec
- Circuit breaker: 5 failures -> 60s cooldown
- Retry with exponential backoff on failures
- Auto-detects rate limit (429) and auth (401/403) errors
- Reads `UPSTOX_API_KEY`, `UPSTOX_API_SECRET`, `UPSTOX_ACCESS_TOKEN` from env

**ZerodhaBroker** - For real trading with Zerodha:
- Kite Connect SDK integration
- Rate limiting: 100 requests/min, 2 requests/sec
- Circuit breaker with fund cache fallback
- Token auto-load from `~/.zerodha_token.json`
- Reads `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_ACCESS_TOKEN` from env
- Sanity validation on prices (rejects 0, NaN, Inf, >1M)
- Partial fill detection and alerting

---

### Utilities (5 files, ~1,340 lines)

| File | Lines | Classes | Purpose |
|------|-------|---------|---------|
| `utils/rate_limiter.py` | 271 | `RateLimiter`, `BrokerRateLimiter` | Prevent API bans |
| `utils/error_handler.py` | 361 | `CircuitBreaker`, exceptions | Error recovery |
| `utils/trade_db.py` | 453 | `TradeDB` | SQLite persistence, crash recovery |
| `utils/alerts.py` | 144 | Telegram functions | Trade notifications |
| `utils/platform.py` | 110 | IST timezone, yfinance timeout | Platform utilities |

**Rate Limiter**:
- Per-minute and per-second request tracking
- Exponential backoff with jitter on failures
- Automatic cooldown when rate-limited (429)
- Broker-specific presets (Upstox: 200 RPM / 15 RPS, Zerodha: 100 RPM / 2 RPS)
- `@rate_limited` decorator for wrapping API calls
- Thread-safe with locks

**Error Handler**:
- `CircuitBreaker`: CLOSED -> OPEN (5 failures) -> HALF_OPEN (after 60s) -> CLOSED (3 successes)
- `@retry_with_backoff`: decorator with configurable retries and exponential delay
- Categorized exceptions: TRANSIENT (retry), RATE_LIMIT (long backoff), FATAL (stop), DATA (skip), AUTH (needs fix)
- `safe_execute()`: wrapper that returns default on failure

**Trade Database** (`utils/trade_db.py`):
- SQLite-backed persistence at `data/trades.db`
- Tables: trades, orders, position_snapshots, daily_summary
- **Crash recovery**: Marks orphaned trades from previous sessions
- **Audit trail**: Full entry/exit records with P&L
- Schema migrations support
- Disk space checking before writes

**Alerts** (`utils/alerts.py`):
- Telegram notifications for critical events
- Alerts: trade entry/exit, kill switch, errors, daily reports, position mismatches
- Non-blocking async sending
- Configurable via `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`

**Platform Utilities** (`utils/platform.py`):
- `now_ist()`, `today_ist()`, `time_ist()` - IST timezone functions
- `yfinance_with_timeout()` - Timeout wrapper for yfinance calls
- `check_disk_space()` - Disk space validation
- `is_pid_running()` - PID validation for daemon

---

### Signal Engine (`src/` - 200+ Python files)

The enhanced stock-predict system provides the intelligence:

```
src/signals/enhanced_generator.py  <- 7-Model Ensemble (default)
src/signals/generator.py           <- Basic single-model (fallback)
|
|-- src/data/                      <- Price data (yfinance), news (NewsAPI, GNews)
|-- src/features/                  <- Technical indicators (RSI, MACD, BB, ATR, VWAP)
|   +-- technical.py               <- TechnicalIndicators.calculate_all()
|-- src/models/                    <- Scoring engines
|   +-- scoring.py                 <- Base ScoringEngine
|   +-- enhanced_scoring.py        <- 7-Model EnhancedScoringEngine
|-- src/physics/                   <- Physics-inspired momentum models
|   +-- physics_engine.py          <- Momentum, spring, energy, network
|-- src/math_models/               <- Fourier, fractal, entropy models
|   +-- math_engine.py             <- Cycles, Hurst, entropy, mechanics
|-- src/ml/                        <- HMM Regime Detection
|   +-- hmm_regime_detector.py     <- Market regime identification
|-- src/macro/                     <- Macro indicators [NEW]
|   +-- macro_engine.py            <- Commodities, currencies, bonds
|-- src/alternative_data/          <- Alternative data [NEW]
|   +-- alternative_engine.py      <- Earnings, options, institutional
|-- src/core/                      <- Advanced math [NEW]
|   +-- advanced_engine.py         <- Kalman, Wavelet, PCA, Markov, DQN
|   +-- advanced_math/             <- Individual advanced algorithms
|-- src/insights/                  <- AI analysis (OpenAI, Anthropic)
|-- src/execution/                 <- Trailing stop, gap handler, re-entry
|   +-- trailing_stop.py           <- TrailingStopCalculator
|-- src/data/news_fetcher.py       <- NewsFetcher
+-- src/storage/                   <- Database, prediction models
```

**Enhanced Pipeline** (`EnhancedSignalGenerator.run()`):
1. Load stock universe (NIFTY 50)
2. Fetch 1 year of price data per stock
3. Calculate all technical indicators
4. Fetch and analyze news (sentiment)
5. **Score each stock with 7-model ensemble:**
   - Base technical + momentum + volume + news
   - Physics engine (momentum conservation, spring reversion)
   - Math engine (Fourier, Hurst, entropy)
   - HMM regime detection (5 market states)
   - Macro engine (commodities, currencies, bonds, semiconductors)
   - Alternative data (earnings, options flow, institutional)
   - Advanced math (Kalman, Wavelet, PCA, Markov, DQN)
6. **Ensemble voting** - require 3+ model agreement
7. Rank by confidence, filter by threshold (>0.60)
8. Generate explanations for top picks
9. Return predictions with entry, SL, target prices

---

### Additional Components

| File | Lines | Purpose |
|------|-------|---------|
| `features/supertrend.py` | 253 | Supertrend indicator (period=7, mult=2.5) |
| `config/trading_config.py` | 343 | Trading system config (reads from .env) |
| `config/settings.py` | 71 | Signal engine config (Pydantic) |
| `config/thresholds.py` | 297 | Scoring engine thresholds |
| `scripts/upstox_auth.py` | 225 | Upstox OAuth2 token helper |
| `scripts/zerodha_auto_login.py` | ~150 | Zerodha auto-login helper |
| `scripts/zerodha_setup.py` | ~100 | Zerodha initial setup |
| `scripts/setup_daemon.sh` | 81 | macOS LaunchAgent installer |
| `scripts/setup_systemd.sh` | ~50 | Linux systemd setup |
| `src/execution/trailing_stop.py` | 251 | Trailing stop calculator (4 methods) |
| `src/data/news_fetcher.py` | 350 | NewsAPI + GNews + RSS fetcher |
| `src/features/news_features.py` | 271 | OpenAI sentiment extraction |
| `src/features/technical.py` | 420 | 20+ technical indicators |

---

## Configuration (.env)

Every parameter is configurable via `.env`. The system reads values at startup with sensible defaults.

### Capital & Risk

```bash
INITIAL_CAPITAL=100000          # Starting capital (Rs.)
HARD_STOP_LOSS=80000            # Kill switch threshold
MAX_DAILY_LOSS_PCT=0.05         # 5% max loss per day -> stops trading
MAX_PER_TRADE_RISK_PCT=0.02     # 2% max risk per single trade
MAX_POSITION_PCT=0.25           # 25% max capital in one stock
BASE_POSITION_PCT=0.15          # Base position size (scaled by confidence)
MAX_POSITIONS=3                 # Max concurrent open positions
TARGET_DAILY_RETURN_PCT=0.03    # 3% daily target
```

### Market Hours

```bash
PRE_MARKET_START=8:45           # Start preparing signals
MARKET_OPEN=9:15                # NSE opens
ENTRY_WINDOW_START=9:30         # Start placing trades
ENTRY_WINDOW_END=14:30          # No new trades after this
SQUARE_OFF_START=14:30          # Begin exiting positions
SQUARE_OFF_TIME=15:00           # Force exit everything
MARKET_CLOSE=15:30              # NSE closes
POST_MARKET_END=16:00           # Post-market reporting
```

### Signal Thresholds

```bash
MIN_CONFIDENCE=0.60             # Only trade 60%+ confidence signals
MIN_RISK_REWARD=1.5             # Only trade 1.5:1 R:R or better
ENABLE_BUY_SIGNALS=true         # BUY signals active
ENABLE_SELL_SIGNALS=false       # SELL disabled (28.6% accuracy)
SIGNAL_CACHE_MINUTES=5          # Cache signals & price data for N minutes
MIN_STOP_DISTANCE_PCT=0.005     # Min 0.5% stop distance
MAX_STOP_DISTANCE_PCT=0.03      # Max 3% stop distance
```

### Strategy Parameters

```bash
SUPERTREND_PERIOD=7             # Supertrend lookback
SUPERTREND_MULTIPLIER=2.5       # Supertrend band width
RSI_PERIOD=14                   # RSI lookback
RSI_OVERSOLD=30                 # RSI oversold level
RSI_OVERBOUGHT=70               # RSI overbought level
ADX_PERIOD=14                   # ADX lookback
ADX_STRONG_TREND=25             # Min ADX for trending market (filter gate)
VOLUME_THRESHOLD=1.5            # Need 1.5x avg volume
TARGET_MULTIPLIER=1.5           # Target = 1.5x risk
TRAILING_START_PCT=0.01         # Start trailing at 1% profit
TRAILING_DISTANCE_PCT=0.005     # Trail 0.5% behind price
```

### Order Execution

```bash
MAX_SLIPPAGE_PCT=0.005          # 0.5% max slippage
ORDER_TIMEOUT_SECONDS=30        # Wait for fill
ORDER_RETRY_COUNT=3             # Retry failed orders
ORDER_RETRY_DELAY=2.0           # Seconds between retries
USE_LIMIT_ORDERS=true           # LIMIT vs MARKET orders
```

### Polling Intervals

```bash
SIGNAL_REFRESH_SECONDS=300      # Refresh signals every 5 min
POSITION_CHECK_SECONDS=60       # Check positions every 1 min
QUOTE_REFRESH_SECONDS=30        # Refresh prices every 30 sec
NEWS_CHECK_SECONDS=300          # Check news for held positions every 5 min
```

### API Keys - Brokers

```bash
# Upstox (for live trading)
UPSTOX_API_KEY=your-key
UPSTOX_API_SECRET=your-secret
UPSTOX_ACCESS_TOKEN=your-token  # Expires daily, refresh via upstox_auth.py

# Zerodha (alternative broker)
ZERODHA_API_KEY=your-key
ZERODHA_API_SECRET=your-secret
ZERODHA_ACCESS_TOKEN=your-token  # Expires daily, refresh via zerodha_auto_login.py
```

### API Keys - AI & News

```bash
# AI & News (required for signal generation)
OPENAI_API_KEY=sk-...
FIRECRAWL_API_KEY=fc-...
GNEWS_API_KEY=...

# Optional
NEWS_API_KEY=...
```

### Notifications

```bash
# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Watchlist (optional override)

```bash
WATCHLIST=RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK
```

Default: 20 NIFTY 50 stocks (defined in config/trading_config.py).

---

## Risk Management

### Kill Switch
```
Portfolio value < HARD_STOP_LOSS -> ALL trading stops permanently
                                   (until manually reset)
```

### Daily Loss Limit
```
Today's P&L < -(INITIAL_CAPITAL x MAX_DAILY_LOSS_PCT) -> No more trades today
```

### Per-Trade Risk
```
Max loss per trade = INITIAL_CAPITAL x MAX_PER_TRADE_RISK_PCT
Position size = Max loss / (Entry price - Stop loss)
```

### Stop-Loss Validation
```
Stop distance must be between MIN_STOP_DISTANCE_PCT and MAX_STOP_DISTANCE_PCT
Too tight -> rejected (noise will trigger it)
Too wide -> rejected (too much risk)
```

### Position Limits
```
Max concurrent positions: MAX_POSITIONS (default 3)
Max capital per position: MAX_POSITION_PCT (default 25%)
```

### Trailing Stop
```
Method: Percentage-based
Activation: After TRAILING_START_PCT (1%) profit
Distance: TRAILING_DISTANCE_PCT (0.5%) behind highest price

Example:
  Entry: 100.00
  Trailing activates at: 101.00 (1% up)
  Price hits 103.00 -> stop moves to 102.48 (0.5% below high)
  Price hits 104.00 -> stop moves to 103.48
  Price drops to 103.48 -> STOP LOSS HIT -> EXIT
```

### SL Order Protection
```
After entering a position:
  1. Place SL order with broker
  2. If SL order FAILS -> EXIT position immediately
  3. No position is ever left without a stop loss
```

---

## Error Handling

### Rate Limiting
```
Request -> Rate Limiter -> API
              |
              +- Under limit -> proceed
              +- At limit -> wait (blocks)
              +- 429 error -> exponential backoff + cooldown
```

Broker-specific limits:
- Upstox: 200 requests/min, 15/sec
- Zerodha: 100 requests/min, 2/sec

### Circuit Breaker
```
CLOSED (normal) --5 failures--> OPEN (blocked)
                                    |
                               60 seconds
                                    |
                                    v
                              HALF_OPEN (testing)
                                    |
                        +-----------+-----------+
                   3 successes    failure
                        |           |
                        v           v
                     CLOSED       OPEN
```

### Order Retry
```
Place order -> Failed?
    +- Yes -> Wait ORDER_RETRY_DELAY -> Retry (up to ORDER_RETRY_COUNT times)
    +- No -> Continue
```

### Price Validation
```
Get LTP from broker -> price is 0 or None?
    +- Yes -> Skip this position check cycle (don't trigger false SL)
    +- No -> Continue with position monitoring
```

### Exit Price Validation
```
Get exit price from broker:
    1. Try order fill price (order_book.average_price)
    2. Fallback: get_ltp(symbol)
    3. Last resort: use entry_price (zero P&L)
    Never calculate P&L with price = 0
```

### Graceful Shutdown
```
Ctrl+C or SIGTERM received
    -> Set shutdown flag
    -> Orchestrator detects flag
    -> Exits all open positions (market orders)
    -> Generates daily report
    -> Exits cleanly
```

---

## File Structure

```
stock-predict/
|-- start.py                    <- RUN THIS EVERY MORNING
|-- trade.py                    <- CLI commands
|-- daemon.py                   <- Background daemon (alternative)
|-- main.py                     <- Enhanced signal generator CLI
|-- .env                        <- Your configuration (gitignored)
|-- .env.example                <- Configuration reference
|
|-- config/
|   |-- trading_config.py       <- Trading system config (reads .env)
|   |-- settings.py             <- Signal engine config (Pydantic)
|   |-- thresholds.py           <- Scoring engine thresholds
|   +-- symbols.json            <- Stock universe
|
|-- agent/
|   |-- orchestrator.py         <- Main trading loop
|   |                              Trade persistence (SQLite)
|   |                              Position reconciliation
|   |                              Crash recovery
|   |                              Stop loss monitoring
|   |                              Trailing stop (connected)
|   |                              News monitoring for positions
|   |                              SL failure protection
|   |                              Telegram alerts
|   +-- signal_adapter.py       <- Predictions -> trade decisions
|                                  Supertrend gate
|                                  ADX filter
|                                  Price cache with expiry
|
|-- broker/
|   |-- base.py                 <- Abstract broker interface
|   |-- paper.py                <- Paper trading (yfinance)
|   |-- upstox.py               <- Real Upstox API
|   +-- zerodha.py              <- Real Zerodha Kite API
|
|-- risk/
|   +-- manager.py              <- Kill switch, position sizing, PnL tracking
|
|-- utils/
|   |-- rate_limiter.py         <- API rate limiting
|   |-- error_handler.py        <- Circuit breaker, retries
|   |-- trade_db.py             <- SQLite persistence, crash recovery
|   |-- alerts.py               <- Telegram notifications
|   +-- platform.py             <- IST timezone, yfinance timeout
|
|-- features/
|   +-- supertrend.py           <- Supertrend indicator
|
|-- src/                        <- Signal prediction engine (200+ files)
|   |-- signals/
|   |   |-- enhanced_generator.py  <- 7-Model Ensemble (default)
|   |   +-- generator.py           <- Basic single-model (fallback)
|   |-- models/
|   |   |-- scoring.py             <- Base ScoringEngine
|   |   +-- enhanced_scoring.py    <- 7-Model ensemble
|   |-- physics/                   <- Physics-inspired models
|   |-- math_models/               <- Math-based models
|   |-- ml/                        <- HMM Regime Detection
|   |-- macro/                     <- Macro indicators (commodities, bonds)
|   |-- alternative_data/          <- Alternative data (earnings, options)
|   |-- core/                      <- Advanced math (Kalman, Wavelet, PCA)
|   |-- data/                      <- Price, news, market data
|   |-- features/                  <- Technical indicators
|   |-- execution/                 <- Trailing stop, gap handler
|   +-- storage/                   <- Database, prediction models
|
|-- scripts/
|   |-- upstox_auth.py          <- Upstox OAuth helper
|   |-- zerodha_auto_login.py   <- Zerodha auto-login helper
|   |-- zerodha_setup.py        <- Zerodha initial setup
|   |-- setup_daemon.sh         <- macOS daemon installer
|   +-- setup_systemd.sh        <- Linux systemd setup
|
|-- logs/                       <- Daily log files
|-- data/                       <- SQLite databases, cache
|-- ARCHITECTURE.md             <- This file
|-- ENHANCED_SYSTEM.md          <- 7-Model ensemble documentation
+-- LEARNING.md                 <- Strategy patterns from reference repos
```

**Core trading system**: 16 files, ~7,200 lines
**Signal engine (src/)**: 200+ Python files (ML/AI system with 7-model ensemble)

---

## Logging

All activity logged to `logs/trading_YYYY-MM-DD.log`:
- Signal generation results (7-model ensemble votes)
- Supertrend/ADX filter decisions
- Every order placed (symbol, qty, price, type)
- Position entries and exits
- Stop loss hits
- Trailing stop movements
- News sentiment alerts for held positions
- P&L per trade
- Risk manager decisions
- Errors and retries
- Daily summary report

**Trade Database** (`data/trades.db`):
- SQLite with WAL mode for crash safety
- Tables: trades, orders, position_snapshots, daily_summary
- Full audit trail with schema migrations
- Crash recovery for orphaned trades

Logs rotate daily, kept for 30 days.

---

## Testing

### Paper Trading (recommended first)
```bash
python start.py          # Uses real market data, fake orders
```
- Real prices from yfinance
- Simulated fills with slippage
- Full risk management active
- Trade persistence to SQLite active
- Trailing stop, news monitoring, all filters active
- Telegram alerts (if configured)
- See how the system would perform without risking money

### Test Connection
```bash
python trade.py test                    # Test paper broker
python trade.py test --live             # Test Upstox connection
python trade.py test --live --broker zerodha  # Test Zerodha connection
```

### View Signals (without trading)
```bash
python trade.py signals                # All signals (7-model ensemble)
python trade.py signals -s RELIANCE    # Single stock

python main.py analyze RELIANCE        # Enhanced single stock analysis
python main.py run --basic             # Basic mode (single model)
```

---

## Edge Cases Handled

| Edge Case | What Happens | File |
|-----------|-------------|------|
| Broker returns price = 0 | Skip position check cycle | orchestrator.py |
| SL order placement fails | Exit position immediately | orchestrator.py |
| get_order_status() returns None | Use fallback price | orchestrator.py |
| Exit price is 0 or None | Fallback chain: LTP -> entry price | orchestrator.py |
| Trailing stop division by zero | Guard: `risk_pct or 0.02` | orchestrator.py |
| DX/ADX division by zero | `replace(0, np.nan)` guards | technical.py |
| Trend filter throws exception | Reject signal (conservative) | signal_adapter.py |
| Price cache stale | Expires with signal_cache_minutes | signal_adapter.py |
| SELL signal + Supertrend | Requires downtrend (-1), not uptrend | signal_adapter.py |
| cancel_order on PENDING | Handled (was only OPEN before) | paper.py |
| on_trade_complete() tracking | Called on every exit | orchestrator.py |
| System crash with open trades | Marked as ORPHANED in DB on restart | trade_db.py |
| Broker position mismatch | Alert sent, manual reconciliation | orchestrator.py |
| SL modify fails | Alert sent, cancel+replace fallback | orchestrator.py |
| Low disk space | Critical alert, DB writes may fail | trade_db.py |

---

## Known Limitations

1. **Broker tokens expire daily** - Must run auth scripts if token expires:
   - Upstox: `python scripts/upstox_auth.py`
   - Zerodha: `python scripts/zerodha_auto_login.py`

2. **SELL signals disabled** - Only 28.6% accuracy (inverted). BUY-only until improved

3. **BUY accuracy target 65-68%** - Enhanced 7-model ensemble targets 65-68% (improved from 57.6%)

4. **NSE holidays hardcoded for 2026** - Need to update annually in broker config files

5. **No intraday candle data** - Uses daily data + yfinance. Would need WebSocket for real-time candles

6. **Paper broker SL orders don't auto-trigger** - SL monitoring is done by orchestrator, not broker simulation

7. **HOLDING phase unreachable with defaults** - `entry_window_end` and `square_off_start` are both 14:30, so the HOLDING phase condition (`14:30 <= now < 14:30`) is never true. Position monitoring still works because `_handle_entry_window()` also calls `_check_positions()`. To enable a dedicated HOLDING phase, set `ENTRY_WINDOW_END` earlier than `SQUARE_OFF_START` in `.env`

---

*Last Updated: February 11, 2026*
*Version: 3.0 - With 7-Model Ensemble, Zerodha Support, Trade Persistence, and Telegram Alerts*
