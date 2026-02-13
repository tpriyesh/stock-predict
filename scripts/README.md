# Trading Scripts

Setup and management scripts for the trading agent.

## Quick Start

### 1. Setup Upstox API

First, get your API credentials from [Upstox Developer](https://developer.upstox.com/):

```bash
python scripts/upstox_auth.py
```

This will:
- Open browser for Upstox login
- Get access token automatically
- Save credentials to `.env`

**Note:** Upstox tokens expire daily. Re-run this script if you get auth errors.

### 2. Test Connection

```bash
# Test paper broker (uses yfinance for prices)
python trade.py test

# Test Upstox connection
python trade.py test --live
```

### 3. Start Trading

**Paper Trading (Safe - No real money):**
```bash
python trade.py start
```

**Live Trading (Real money):**
```bash
python trade.py start --live
```

### 4. Run as Daemon (Auto-start daily)

**Option A: Manual daemon**
```bash
python daemon.py run       # Foreground (for testing)
python daemon.py start     # Background
python daemon.py stop      # Stop
python daemon.py status    # Check status
```

**Option B: macOS LaunchAgent (auto-start on login)**
```bash
./scripts/setup_daemon.sh
```

This will:
- Install daemon to run on login
- Configure low CPU priority
- Auto-restart if it crashes

## Files

| File | Purpose |
|------|---------|
| `upstox_auth.py` | Get Upstox OAuth access token |
| `setup_daemon.sh` | Install macOS daemon |
| `com.trading.daemon.plist` | macOS LaunchAgent config |

## Daemon Behavior

The daemon automatically:

1. **Sleeps when market closed** - Uses minimal CPU
2. **Wakes at 8:45 AM** - Pre-market preparation
3. **Trades 9:30-10:00 AM** - Entry window
4. **Monitors 10:00-14:30** - Manage positions
5. **Exits at 14:30-15:00** - Square off all
6. **Sleeps until next day** - After 4:00 PM

On weekends and NSE holidays, sleeps until next trading day.

## Logs

All logs are in `/logs/`:

- `daemon_YYYY-MM-DD.log` - Daemon activity
- `trading_YYYY-MM-DD.log` - Trading activity
- `daemon_stdout.log` - LaunchAgent stdout
- `daemon_stderr.log` - LaunchAgent stderr
