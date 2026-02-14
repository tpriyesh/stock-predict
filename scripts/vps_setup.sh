#!/bin/bash
# VPS Setup Script for Stock Trading Agent
# Run as: sudo bash scripts/vps_setup.sh

set -e

echo "=============================================="
echo "Stock Trading Agent - VPS Setup"
echo "=============================================="
echo ""

# Configuration
USER_NAME="${1:-trader}"
WORK_DIR="${2:-/home/$USER_NAME/stock-predict}"

echo "Setting up for user: $USER_NAME"
echo "Working directory: $WORK_DIR"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Create user if doesn't exist
if ! id "$USER_NAME" &>/dev/null; then
    echo "Creating user: $USER_NAME"
    useradd -m -s /bin/bash "$USER_NAME"
fi

# Install dependencies
echo "Installing dependencies..."
apt-get update
apt-get install -y \
    python3-pip \
    python3-venv \
    sqlite3 \
    logrotate \
    curl \
    cron \
    htop \
    vim

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$WORK_DIR"/{data,logs,backups}
chown -R "$USER_NAME:$USER_NAME" "$WORK_DIR"

# Setup logrotate
echo "Configuring log rotation..."
cat > /etc/logrotate.d/trading-agent << EOF
$WORK_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER_NAME $USER_NAME
}
EOF

# Setup systemd service
echo "Creating systemd service..."
cat > /etc/systemd/system/trading-agent.service << EOF
[Unit]
Description=Stock Trading Agent
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$WORK_DIR
Environment="PATH=$WORK_DIR/venv/bin"
EnvironmentFile=$WORK_DIR/.env
ExecStart=$WORK_DIR/venv/bin/python $WORK_DIR/start.py --live
ExecStop=$WORK_DIR/venv/bin/python $WORK_DIR/trade.py stop
Restart=on-failure
RestartSec=60
StartLimitInterval=300
StartLimitBurst=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trading-agent

[Install]
WantedBy=multi-user.target
EOF

# Setup disk check cron
echo "Setting up disk monitoring..."
cat > /usr/local/bin/trading-disk-check.sh << 'DISKSCRIPT'
#!/bin/bash
WORK_DIR="##WORK_DIR##"
MIN_FREE_GB=5
ALERT_FILE="/tmp/disk_alert_sent"
TODAY=$(date +%Y%m%d)

FREE_KB=$(df "$WORK_DIR" | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))

if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    logger "Trading Agent: Low disk space - ${FREE_GB}GB free"
    
    if [ -f "$ALERT_FILE" ] && [ "$(cat $ALERT_FILE)" = "$TODAY" ]; then
        exit 0
    fi
    
    if [ -f "$WORK_DIR/.env" ]; then
        export $(grep -E '^TELEGRAM_' "$WORK_DIR/.env" | xargs)
        if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d "chat_id=${TELEGRAM_CHAT_ID}" \
                -d "text=⚠️ DISK SPACE WARNING%0AFree: ${FREE_GB}GB (min: ${MIN_FREE_GB}GB)%0APath: $WORK_DIR" > /dev/null
        fi
    fi
    
    echo "$TODAY" > "$ALERT_FILE"
    find "$WORK_DIR/logs" -name "*.log" -mtime +30 -delete 2>/dev/null
    find "$WORK_DIR/backups" -name "*.db" -mtime +30 -delete 2>/dev/null
else
    rm -f "$ALERT_FILE"
fi
DISKSCRIPT

sed -i "s|##WORK_DIR##|$WORK_DIR|g" /usr/local/bin/trading-disk-check.sh
chmod +x /usr/local/bin/trading-disk-check.sh

# Setup cron jobs
echo "Setting up cron jobs..."
CRON_FILE="/tmp/trading-cron"
cat > "$CRON_FILE" << EOF
# Trading Agent Cron Jobs

# Disk check every 30 minutes
*/30 * * * * /usr/local/bin/trading-disk-check.sh

# Daily database backup at 6 PM IST (12:30 PM UTC)
30 12 * * * sqlite3 $WORK_DIR/data/trades.db ".backup $WORK_DIR/backups/trades_\$(date +\%Y\%m\%d).db"

# Cleanup old backups (keep 30 days)
0 13 * * * find $WORK_DIR/backups -name "*.db" -mtime +30 -delete

# Token refresh reminder at 8:30 AM IST (3:00 AM UTC)
0 3 * * 1-5 echo "Refresh broker token today" | logger
EOF

chown "$USER_NAME:$USER_NAME" "$CRON_FILE"
crontab -u "$USER_NAME" "$CRON_FILE"
rm -f "$CRON_FILE"

# Setup log directory permissions
mkdir -p /var/log/trading-agent
chown "$USER_NAME:$USER_NAME" /var/log/trading-agent

# Enable systemd service
systemctl daemon-reload
systemctl enable trading-agent.service

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy your code to: $WORK_DIR"
echo "2. Create virtualenv: python3 -m venv $WORK_DIR/venv"
echo "3. Install requirements: pip install -r requirements.txt"
echo "4. Configure .env file: cp .env.example .env"
echo "5. Set ownership: chown -R $USER_NAME:$USER_NAME $WORK_DIR"
echo "6. Test in paper mode: python start.py"
echo "7. Start live trading: systemctl start trading-agent"
echo ""
echo "Useful commands:"
echo "  systemctl start trading-agent   # Start trading"
echo "  systemctl stop trading-agent    # Stop trading"
echo "  systemctl status trading-agent  # Check status"
echo "  journalctl -u trading-agent -f  # View logs"
echo "  python trade.py status          # Portfolio status"
echo ""
echo "REMINDERS:"
echo "  - Refresh broker token DAILY before 9:15 AM IST"
echo "  - Update NSE holidays annually (Dec 2026)"
echo "  - Monitor Telegram alerts"
echo ""
