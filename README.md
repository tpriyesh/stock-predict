# Stock Prediction & Trading Agent

Deterministic, evidence-based stock prediction and automated trading system for Indian markets (NSE).

## Features

### Signal Generation (7-Model Ensemble)
- **7-Model Ensemble Scoring** - Combines 7 independent prediction models for robust signals
- **Top 10 Daily Picks** for intraday and swing trading
- **Full Reasoning Lineage** - every recommendation shows WHY with traceable evidence
- **Multi-Model Agreement** - Requires 3+ models to agree for a signal (configurable)
- **Explainable Predictions** - technical + news + market context + macro indicators

### Automated Trading
- **Dual Broker Support** - Upstox and Zerodha (Kite Connect)
- **Paper Trading** - Test with simulated orders before going live
- **Risk Management** - Kill switch, daily loss limits, position sizing
- **Trade Persistence** - SQLite database for audit trail and crash recovery
- **Telegram Alerts** - Real-time notifications for trades and errors

## Quick Start

### 1. Install Dependencies

```bash
cd stock-predict
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- `OPENAI_API_KEY` - News sentiment analysis
- `GNEWS_API_KEY` - News headlines
- Broker credentials (choose one):
  - `UPSTOX_API_KEY`, `UPSTOX_API_SECRET` - For Upstox trading
  - `ZERODHA_API_KEY`, `ZERODHA_API_SECRET` - For Zerodha trading

Optional:
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` - Trade notifications

### 3. Run Analysis

```bash
# Enhanced 7-model ensemble analysis (default)
python main.py analyze RELIANCE
python main.py run

# Run for specific stocks
python main.py run --symbols TCS,INFY,HDFCBANK

# Require 4+ model agreement (stricter)
python main.py run --min-agreement 4

# Basic mode (single model - faster but less accurate)
python main.py run --basic

# Launch dashboard
python main.py ui
```

### 4. Start Trading

```bash
# Paper trading (recommended first)
python start.py

# Live trading with Upstox
python start.py --live

# Live trading with Zerodha
python start.py --live --broker zerodha

# CLI commands
python trade.py start --live
python trade.py status
python trade.py positions
python trade.py pnl
```

## Output Example

```
ANALYSIS: RELIANCE (INTRADAY) [ENHANCED]
============================================================

Signal: BUY (73% confidence)

Model Agreement: 6/7 models
Signal Strength: STRONG
Market Regime: TRENDING_BULL (stability: 85%)
Predictability: 78%
Strategy: trend_following
Sector: Oil & Gas

Model Votes:
  [+] Base: BUY
  [+] Physics: BUY
  [+] Math: BUY
  [+] Hmm Regime: BUY
  [+] Macro: BUY
  [+] Alternative: HOLD
  [+] Advanced: BUY

Trade Plan:
  Entry:  ₹2,450
  Stop:   ₹2,410
  Target: ₹2,530
  Risk/Reward: 2.0:1

Key Signals:
  - [TECHNICAL] RSI oversold bounce from 28 → 35
  - [PHYSICS] Momentum conservation indicates upward pressure
  - [MATH] Fourier cycle alignment - bullish phase
  - [REGIME] TRENDING_BULL detected (stable)
  - [MACRO] Crude oil decline favorable for refiners
  - [NEWS] Q3 results beat estimates (source: MoneyControl)
  
Warnings:
  - High volatility period detected

============================================================
```

## 7-Model Ensemble Architecture

The enhanced signal engine combines 7 independent models:

| Model | Weight | Description |
|-------|--------|-------------|
| **Base Technical** | 25% | RSI, MACD, Bollinger, Momentum, Volume, News |
| **Physics Engine** | 18% | Momentum conservation, spring reversion, energy |
| **Math Engine** | 14% | Fourier cycles, Hurst exponent, entropy |
| **HMM Regime** | 13% | Hidden Markov Model - 5 market states |
| **Macro Engine** | 10% | Commodities, currencies, bonds, semiconductors |
| **Alternative Data** | 10% | Earnings, options flow, institutional flow |
| **Advanced Math** | 10% | Kalman, Wavelet, PCA, Markov, DQN |

**Multi-Model Agreement:**
- 5-7 models agree → STRONG signal (HIGH conviction)
- 3-4 models agree → MODERATE signal (standard trade)
- 1-2 models agree → Filtered out (too weak)

**SELL signals disabled** - Historical accuracy was 28.6% (inverted).

## Project Structure

```
stock-predict/
├── config/                  # Configuration
│   ├── trading_config.py    # Trading system config
│   ├── settings.py          # Signal engine config
│   └── thresholds.py        # Scoring thresholds
│
├── src/                     # Signal Prediction Engine
│   ├── signals/
│   │   ├── enhanced_generator.py  # 7-Model Ensemble (default)
│   │   └── generator.py           # Basic single-model
│   ├── models/
│   │   ├── enhanced_scoring.py    # 7-Model ensemble logic
│   │   └── scoring.py             # Base scoring
│   ├── physics/             # Physics-inspired models
│   ├── math_models/         # Mathematical models
│   ├── ml/                  # HMM Regime Detection
│   ├── macro/               # Macro indicators (NEW)
│   ├── alternative_data/    # Alternative data (NEW)
│   ├── core/                # Advanced math (NEW)
│   ├── data/                # Price, news fetchers
│   ├── features/            # Technical indicators
│   └── storage/             # Database models
│
├── agent/                   # Trading Agent
│   ├── orchestrator.py      # Main trading loop
│   └── signal_adapter.py    # Signal filtering
│
├── broker/                  # Broker Integrations
│   ├── base.py              # Abstract interface
│   ├── paper.py             # Paper trading
│   ├── upstox.py            # Upstox API
│   └── zerodha.py           # Zerodha Kite API
│
├── risk/                    # Risk Management
│   └── manager.py           # Kill switch, position sizing
│
├── utils/                   # Utilities
│   ├── trade_db.py          # SQLite persistence
│   ├── alerts.py            # Telegram notifications
│   ├── rate_limiter.py      # API rate limiting
│   ├── error_handler.py     # Circuit breakers
│   └── platform.py          # IST timezone, timeouts
│
├── ui/                      # Streamlit dashboard
├── scripts/                 # Setup scripts
│   ├── upstox_auth.py
│   ├── zerodha_auto_login.py
│   └── setup_daemon.sh
│
├── data/                    # SQLite databases
├── logs/                    # Daily log files
│
├── start.py                 # Trading entry point
├── trade.py                 # CLI commands
├── daemon.py                # Background daemon
└── main.py                  # Signal generator CLI
```

## How It Works

### Signal Generation Pipeline

1. **Data Ingestion**: 
   - Fetch OHLCV from Yahoo Finance
   - News from NewsAPI, GNews, RSS
   - Macro data (commodities, currencies, bonds)

2. **Feature Engineering**: 
   - Calculate 20+ technical indicators
   - Extract news sentiment via OpenAI
   - Compute physics, math, and regime features

3. **7-Model Ensemble Scoring**:
   - Run all 7 models independently
   - Each model votes BUY/HOLD/SELL
   - Count agreements, calculate weighted score

4. **Filtering**: 
   - Require min model agreement (default: 3)
   - Apply regime filters (skip CHOPPY markets)
   - Liquidity, volatility, diversification filters

5. **Trade Execution** (if enabled):
   - Position sizing based on risk
   - Immediate stop-loss placement
   - Trailing stop management
   - News monitoring for held positions

### Risk Controls

| Control | Description |
|---------|-------------|
| **Kill Switch** | Stop ALL trading if portfolio < threshold |
| **Daily Loss Limit** | Max 5% loss per day |
| **Per-Trade Risk** | Max 2% capital at risk per trade |
| **Position Limits** | Max 3 concurrent positions, 25% per position |
| **Liquidity Filter** | Only stocks with >₹10Cr daily volume |
| **Volatility Cap** | Skip stocks with ATR >5% |
| **Sector Diversification** | Max 3 stocks per sector |
| **Confidence Threshold** | Only trade >60% confidence |

## API Keys Required

| API | Purpose | Free Tier |
|-----|---------|-----------|
| OpenAI | News sentiment analysis | $5 credit |
| GNews | News headlines | 100 req/day |
| Upstox/Zerodha | Live trading | Broker account |
| Telegram (optional) | Notifications | Free |

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Full system architecture
- **[ENHANCED_SYSTEM.md](ENHANCED_SYSTEM.md)** - 7-Model ensemble details
- **[LEARNING.md](LEARNING.md)** - Strategy patterns from research

## Expected Performance

Based on backtest analysis:

| Mode | Accuracy | Signals | Quality |
|------|----------|---------|---------|
| Basic | 57.6% | More | Lower |
| Enhanced (3+ agreement) | 62-65% | Moderate | High |
| Enhanced (5+ agreement) | 68%+ | Few | Highest |

## Known Limitations

1. **Broker tokens expire daily** - Must refresh daily
2. **SELL signals disabled** - Only BUY/HOLD until SELL model fixed
3. **No real-time WebSocket** - Uses polling (yfinance)
4. **NSE holidays hardcoded** - Update annually

## Disclaimer

This is for educational and research purposes only. Not financial advice.
Always do your own research before trading. Past performance does not
guarantee future results.

---

*Last Updated: February 11, 2026*
