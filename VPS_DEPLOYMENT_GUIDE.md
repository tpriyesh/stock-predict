# VPS Deployment Guide - Stock Trading Agent

Complete guide for deploying the trading agent on a VPS for personal use.

## Quick Start

```bash
# 1. Clone/copy code to VPS
git clone <your-repo> /home/trader/stock-predict
cd /home/trader/stock-predict

# 2. Run automated setup
sudo bash scripts/vps_setup.sh trader /home/trader/stock-predict

# 3. Create virtualenv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Test in paper mode
python start.py

# 6. Start live trading
sudo systemctl start trading-agent
```

## Architecture Review Summary

### What's Production Ready ✓

**Order Execution**
- No double-order risk (no retry on placement)
- Partial fill detection and handling
- Cancel verification before proceeding
- Position exits if SL placement fails

**Stop Loss Protection**
- SL stuck detection (30s grace → market sell)
- SL modify verification with cancel+replace fallback
- Real ATR for trailing stops
- Price validation (rejects 0, NaN, Inf, >20% moves)

**Crash Recovery**
- SQLite persistence with WAL mode
- Orphaned trade detection on startup
- Position reconciliation (broker vs internal)
- 120s timeout for graceful shutdown

**Risk Management**
- Kill switch (portfolio < hard_stop)
- Unrealized P&L included in calculations
- Daily loss limit tracking
- Gap-down and circuit breaker protection

### What Needs Manual Attention ⚠

| Item | Frequency | Action |
|------|-----------|--------|
| Broker token refresh | Daily | Run auth script before 9:15 AM |
| Holiday list update | Annual | Update NSE_HOLIDAYS_2026 → 2027 |
| Database backup | Daily | Automated via cron |
| Log rotation | Daily | Automated via logrotate |
| Disk space check | Every 30 min | Automated via cron |

## Systemd Service Management

```bash
# Start trading
sudo systemctl start trading-agent

# Stop trading (exits all positions first)
sudo systemctl stop trading-agent

# View status
sudo systemctl status trading-agent

# View logs
sudo journalctl -u trading-agent -f

# Restart after config changes
sudo systemctl restart trading-agent

# Disable auto-start
sudo systemctl disable trading-agent
```

## Daily Operations

### Morning Routine (Before 9:15 AM IST)

1. **Refresh broker token** (required daily):
   ```bash
   # For Upstox
   python scripts/upstox_auth.py
   
   # For Zerodha
   python scripts/zerodha_auto_login.py
   ```

2. **Verify Telegram alerts**:
   ```bash
   python -c "from utils.alerts import alert_startup; alert_startup('test', 100000)"
   ```

3. **Check system status**:
   ```bash
   python trade.py status
   ```

### Evening Review (After 3:30 PM)

1. **View daily report**:
   ```bash
   cat data/reports/$(date +%Y-%m-%d).txt
   ```

2. **Check for issues**:
   ```bash
   grep -i "error\|critical\|warning" logs/trading_$(date +%Y-%m-%d).log
   ```

3. **Backup database** (automated, but verify):
   ```bash
   ls -lh backups/
   ```

## Monitoring & Alerts

### Telegram Alerts You'll Receive

| Alert Type | Trigger | Action Needed |
|------------|---------|---------------|
| Trade Entry | Position opened | None (informational) |
| Trade Exit | Position closed | Review P&L |
| Kill Switch | Portfolio < hard_stop | **STOP - Manual intervention** |
| SL Stuck | SL order not triggering | **Check broker** |
| Token Expired | Auth failure | Refresh token immediately |
| Disk Space | < 5GB free | Cleanup or expand disk |
| DB Write Failed | SQLite error | **Check disk space** |

### Log Files

| File | Purpose | Rotation |
|------|---------|----------|
| `logs/trading_YYYY-MM-DD.log` | Main trading log | Daily, 30 days |
| `logs/issues_YYYY-MM-DD.log` | Error/warning only | Daily, 30 days |
| `data/trades.db` | Trade database | Daily backup |
| `data/reports/YYYY-MM-DD.txt` | Human-readable report | 30 days |

## Troubleshooting

### System Won't Start

```bash
# Check config validation
python -c "from config.trading_config import CONFIG; CONFIG.print_summary()"

# Check for missing .env
cat .env | grep -E "API_KEY|TOKEN"

# Check permissions
ls -la data/ logs/

# View detailed error
sudo journalctl -u trading-agent --no-pager -n 100
```

### Token Expired

```bash
# System will auto-stop if token expires
# To fix:
python scripts/upstox_auth.py  # or zerodha_auto_login.py
sudo systemctl restart trading-agent
```

### Database Issues

```bash
# Check database integrity
sqlite3 data/trades.db "PRAGMA integrity_check;"

# View recent trades
sqlite3 data/trades.db "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 5;"

# Check for orphaned trades
sqlite3 data/trades.db "SELECT COUNT(*) FROM trades WHERE status='OPEN';"
```

### Position Stuck (Not Exiting)

```bash
# Check position status
python trade.py positions

# Force emergency exit
python -c "
from agent.orchestrator import TradingOrchestrator
from broker import create_broker
from risk.manager import RiskManager

broker = create_broker()
rm = RiskManager(broker)
orch = TradingOrchestrator(broker, rm)
orch._emergency_exit_all('Manual intervention')
"
```

## Security Checklist

- [ ] VPS firewall enabled (only SSH + HTTPS)
- [ ] SSH key authentication only (no passwords)
- [ ] .env file has restrictive permissions (chmod 600)
- [ ] Database directory has correct ownership
- [ ] Non-root user for trading process
- [ ] Regular security updates (`apt-get update && apt-get upgrade`)

## Performance Tuning

### For Low-End VPS (1 CPU, 1GB RAM)

```bash
# Reduce signal refresh frequency
# Edit .env:
SIGNAL_REFRESH_SECONDS=600  # Instead of 300

# Reduce watchlist size
WATCHLIST=RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK

# Disable news if needed (not recommended)
# But system will work without it
```

### Database Maintenance

```bash
# Weekly vacuum to optimize
sqlite3 data/trades.db "VACUUM;"

# Check size
ls -lh data/trades.db

# Archive old data if needed
# (Keeps last 90 days in main DB)
```

## Backup Strategy

### Automated (Via Cron)

- Database: Daily at 6 PM IST → `backups/trades_YYYYMMDD.db`
- Retention: 30 days

### Manual Backup

```bash
# Before major changes
sqlite3 data/trades.db ".backup backups/pre_update_$(date +%Y%m%d_%H%M).db"

# Copy to remote
scp data/trades.db user@backup-server:~/backups/
```

## Rollback Procedure

If something goes wrong:

```bash
# 1. Stop trading immediately
sudo systemctl stop trading-agent

# 2. Check for open positions
python trade.py positions

# 3. Manually exit if needed (via broker app)

# 4. Restore database if corrupted
cp backups/trades_20240214.db data/trades.db

# 5. Review logs
tail -n 500 logs/trading_$(date +%Y-%m-%d).log

# 6. Restart in paper mode first
python start.py  # Test without live trading

# 7. Go live again
sudo systemctl start trading-agent
```

## Important Reminders

### ⚠️ Critical (Will Stop Trading)

1. **Daily Token Refresh**: System will NOT trade without valid token
2. **Holiday Updates**: System may try to trade on holidays if not updated
3. **Kill Switch**: Once triggered, manual reset required

### ℹ️ Good to Know

- System auto-exits all positions on shutdown (Ctrl+C or stop)
- Paper mode uses real market data but fake orders
- SQLite database survives crashes (WAL mode)
- Telegram alerts fail silently (check logs if missing)
- Position reconciliation runs every 2 minutes

## Support & Debugging

### Get System Status

```bash
python trade.py status
python trade.py positions
python trade.py pnl
```

### Health Check Script

```bash
# Create health check
cat > /usr/local/bin/trading-health.sh << 'EOF'
#!/bin/bash
WORK_DIR="/home/trader/stock-predict"
cd "$WORK_DIR"

echo "=== Trading Agent Health Check ==="
echo "Time: $(date)"
echo ""

echo "Systemd Status:"
systemctl is-active trading-agent

echo ""
echo "Disk Space:"
df -h "$WORK_DIR" | tail -1

echo ""
echo "Recent Trades:"
sqlite3 data/trades.db "SELECT symbol, status, COUNT(*) FROM trades WHERE entry_time > datetime('now', '-1 day') GROUP BY symbol, status;" 2>/dev/null || echo "DB query failed"

echo ""
echo "Open Positions:"
sqlite3 data/trades.db "SELECT symbol, quantity, entry_price FROM trades WHERE status='OPEN';" 2>/dev/null || echo "No open positions"

echo ""
echo "Recent Errors:"
grep -i "error\|critical" logs/trading_$(date +%Y-%m-%d).log 2>/dev/null | tail -5 || echo "No errors found"

echo ""
echo "=== End Health Check ==="
EOF

chmod +x /usr/local/bin/trading-health.sh
trading-health.sh
```

---

**Last Updated**: February 14, 2026  
**Version**: 3.0
