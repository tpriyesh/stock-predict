# Stock-Predict Trading Agent - User Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Paper Trading (Automated)](#paper-trading-automated)
- [Live Trading (Manual)](#live-trading-manual)
- [Viewing Reports](#viewing-reports)
- [Daemon Management](#daemon-management)
- [Logs & Debugging](#logs--debugging)
- [Configuration](#configuration)
- [Deployment Options](#deployment-options)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your config
cp .env.example .env
# Edit .env with your API keys (OpenAI, Telegram, broker credentials)

# 3. Start automated paper trading (one-time setup)
bash scripts/setup_macos.sh install

# 4. Check reports each evening
python trade.py report
```

That's it. The daemon runs in the background automatically.

---

## Paper Trading (Automated)

Paper trading uses **simulated money** (Rs.1,00,000 starting capital) with **real market data**. No real orders are placed. This is how you validate the strategy before risking real money.

### How It Works

1. **Daemon starts automatically** when you log into your Mac
2. **Before market (8:45 AM)** - Wakes up, generates trading signals using AI + technical analysis
3. **Market hours (9:15 AM - 3:30 PM)** - Executes paper trades based on signals
   - Buys stocks that pass confidence, risk-reward, volume, and trend filters
   - Monitors stop-losses and targets in real-time
   - Exits positions at target, stop-loss, or market close
4. **After market (3:30 PM)** - Generates daily report, saves to database, sends Telegram alert
5. **Overnight** - Sleeps with near-zero CPU usage until next trading day
6. **Weekends & holidays** - Automatically skips non-trading days

### You Don't Need To Do Anything

Once installed, the daemon handles everything. Just check your report in the evening:

```bash
python trade.py report
```

---

## Live Trading (Manual)

Live trading uses **real money** through your broker (Zerodha or Upstox). It is **never automated** - you must start it manually and confirm every time.

### Why Manual?

Safety. The daemon is hardcoded to paper mode. Live trading requires your explicit presence and confirmation to prevent accidental real-money trades.

### How to Run Live Trading

```bash
# Step 1: Stop the paper daemon (to avoid conflicts)
bash scripts/setup_macos.sh stop

# Step 2: Start live trading (choose one)
python start.py --zerodha        # Zerodha (auto-login via TOTP)
python start.py --live            # Uses BROKER env var (Upstox or Zerodha)

# Step 3: Type "I CONFIRM" when prompted
# The agent will trade with real money until market close

# Step 4: After market close, restart paper daemon
bash scripts/setup_macos.sh start
```

### First-Time Broker Setup

**Zerodha:**
```bash
python scripts/zerodha_setup.py   # One-time setup (API key, secret, TOTP)
python trade.py test --zerodha    # Test connection
```

**Upstox:**
```bash
python scripts/upstox_auth.py     # Get access token (expires daily)
python trade.py test --live       # Test connection
```

### Live vs Paper Comparison

| Feature | Paper Trading | Live Trading |
|---------|--------------|--------------|
| Money | Simulated (Rs.1L) | Real money |
| Start method | Automatic (daemon) | Manual (`python start.py --live`) |
| Confirmation | None needed | Must type "I CONFIRM" |
| Broker | PaperBroker (built-in) | Zerodha / Upstox |
| Orders | Simulated fills | Real market orders |
| Risk | Zero | Real capital at risk |
| Recommended for | Strategy validation | After 2+ weeks of good paper results |

---

## Viewing Reports

### Daily Report (Today)

```bash
python trade.py report
```

Shows:
- Daemon status (running or not)
- Today's trades, win rate, P&L
- Per-trade breakdown (symbol, entry, exit, P&L)
- Last 5 trading days performance

### Trade History (Custom Range)

```bash
python trade.py history              # Last 7 days
python trade.py history --days 30    # Last 30 days
```

Shows daily summary: trades, win rate, P&L, portfolio value.

### Today's P&L (Live Broker)

```bash
python trade.py pnl                  # Paper broker
python trade.py pnl --live           # Live broker (Upstox)
python trade.py pnl --zerodha        # Live broker (Zerodha)
```

### Report Files

Daily reports are also saved as text files:
```
data/reports/2026-02-12.txt
data/reports/2026-02-13.txt
...
```

You can open these in any text editor.

### Telegram Alerts

If configured in `.env`, you'll get automatic Telegram messages:
- Daily report (trades, P&L, portfolio value)
- Error alerts (broker issues, kill switch triggers)
- Trade notifications (entry/exit)

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Database (Advanced)

All trades are stored in `data/trades.db` (SQLite). You can query it directly:
```bash
sqlite3 data/trades.db "SELECT * FROM daily_summary ORDER BY trade_date DESC LIMIT 10;"
```

---

## Daemon Management

### Install (First Time)

```bash
bash scripts/setup_macos.sh install
```
- Copies LaunchAgent to `~/Library/LaunchAgents/`
- Starts the daemon immediately
- Auto-starts on every future login

### Check Status

```bash
bash scripts/setup_macos.sh status
```
Shows: running/stopped, PID, recent logs, today's report.

### Start / Stop

```bash
bash scripts/setup_macos.sh stop     # Stop the daemon
bash scripts/setup_macos.sh start    # Start again
```

### Watch Live Logs

```bash
bash scripts/setup_macos.sh logs
```
Tails today's daemon log in real-time. Press Ctrl+C to stop watching.

### Uninstall Completely

```bash
bash scripts/setup_macos.sh uninstall
```
Stops the daemon and removes the LaunchAgent. Paper trading will no longer auto-start.

### What Happens When...

| Scenario | Behavior |
|----------|----------|
| Laptop sleeps | Daemon sleeps too, resumes on wake |
| Laptop restarts | Daemon auto-starts on login |
| Daemon crashes | macOS restarts it after 60 seconds |
| Weekend / holiday | Daemon detects it, sleeps until next trading day |
| Market closed | Daemon sleeps until 8:45 AM next trading day |
| Internet goes down | Daemon retries, logs errors, continues when back |

---

## Logs & Debugging

### Log Locations

| Log | Path | Content |
|-----|------|---------|
| Daemon log | `logs/daemon_YYYY-MM-DD.log` | Daemon lifecycle, sleep/wake |
| Trading log | `logs/trading_YYYY-MM-DD.log` | Trades, signals, errors (DEBUG level) |
| Daemon stdout | `logs/daemon_stdout.log` | LaunchAgent stdout capture |
| Daemon stderr | `logs/daemon_stderr.log` | LaunchAgent stderr capture |
| Daily reports | `data/reports/YYYY-MM-DD.txt` | Human-readable daily summary |

### Useful Commands

```bash
# View today's daemon log
cat logs/daemon_$(date +%Y-%m-%d).log

# View today's trading log (detailed)
cat logs/trading_$(date +%Y-%m-%d).log

# Search for errors in today's log
grep -i error logs/trading_$(date +%Y-%m-%d).log

# Check database for open trades
python trade.py positions

# Test broker connection
python trade.py test
python trade.py test --zerodha

# Generate signals manually (without trading)
python trade.py signals
python trade.py signals -s RELIANCE    # Specific stock
```

### Log Retention

Logs are automatically rotated daily and deleted after 30 days.

---

## Configuration

All settings are in `.env` (copy from `.env.example`).

### Key Settings

```env
# Trading mode (daemon forces "paper" regardless)
TRADING_MODE=paper

# Capital & Risk
INITIAL_CAPITAL=100000
HARD_STOP_LOSS=80000
MAX_DAILY_LOSS_PCT=0.05
MAX_RISK_PER_TRADE=0.02
MAX_POSITIONS=5

# Signal Quality (higher = fewer but better trades)
MIN_CONFIDENCE=0.65
MIN_RISK_REWARD=1.8
ADX_STRONG_TREND=25

# Market Hours (IST)
MARKET_OPEN=09:15
MARKET_CLOSE=15:30
ENTRY_START=09:30
ENTRY_END=14:30

# Broker (for live trading)
BROKER=zerodha
ZERODHA_API_KEY=your_key
ZERODHA_API_SECRET=your_secret
ZERODHA_TOTP_KEY=your_totp_key

# Notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# AI (for signal generation)
OPENAI_API_KEY=your_openai_key
```

---

## Deployment Options

### macOS (Current Setup)

```bash
bash scripts/setup_macos.sh install
```
Uses macOS LaunchAgent. Auto-starts on login.

### Linux VPS (Hostinger, DigitalOcean, etc.)

```bash
bash scripts/setup_systemd.sh
sudo systemctl start stock-predict
sudo systemctl enable stock-predict    # Auto-start on boot
```

### Docker

```bash
# Paper trading
docker compose up -d

# Live trading (Zerodha)
docker compose -f docker-compose.yml -f docker-compose.live.yml up -d

# View logs
docker logs -f stock-predict-trading-agent-1
```

---

## Recommended Workflow

1. **Week 1-2**: Run paper trading (daemon), check `python trade.py report` daily
2. **Evaluate**: Look at win rate, P&L, and trade quality
3. **If win rate > 55% and consistent profit**: Consider live trading
4. **Live trading**: Stop daemon, run `python start.py --zerodha`, confirm manually
5. **Monitor**: Check reports, Telegram alerts, and logs during live trading
