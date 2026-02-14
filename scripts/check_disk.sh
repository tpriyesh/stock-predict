#!/bin/bash
# Disk space monitoring for Trading Agent
# Add to crontab: */30 * * * * /path/to/stock-predict/scripts/check_disk.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"

# Config
MIN_FREE_GB=5
DB_PATH="$WORK_DIR/data/trades.db"
ALERT_FILE="/tmp/disk_alert_sent"

# Get free space
FREE_KB=$(df "$WORK_DIR" | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))

# Check disk space
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "$(date): WARNING - Low disk space: ${FREE_GB}GB free (min: ${MIN_FREE_GB}GB)"
    
    # Send alert only once per day
    TODAY=$(date +%Y%m%d)
    if [ -f "$ALERT_FILE" ]; then
        LAST_ALERT=$(cat "$ALERT_FILE")
        if [ "$LAST_ALERT" = "$TODAY" ]; then
            exit 0
        fi
    fi
    
    # Send Telegram alert if configured
    if [ -f "$WORK_DIR/.env" ]; then
        source "$WORK_DIR/.env"
        if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
            MESSAGE="⚠️ DISK SPACE WARNING%0A"
            MESSAGE+="Free: ${FREE_GB}GB (min: ${MIN_FREE_GB}GB)%0A"
            MESSAGE+="Path: $WORK_DIR%0A"
            MESSAGE+="Time: $(date '+%H:%M:%S')"
            
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d "chat_id=${TELEGRAM_CHAT_ID}" \
                -d "text=${MESSAGE}" \
                -d "parse_mode=HTML" > /dev/null 2>&1
        fi
    fi
    
    echo "$TODAY" > "$ALERT_FILE"
    
    # Cleanup old logs to free space
    echo "$(date): Cleaning old logs..."
    find "$WORK_DIR/logs" -name "*.log" -type f -mtime +30 -delete 2>/dev/null
    
    # Cleanup old backups
    find "$WORK_DIR/backups" -name "*.db" -type f -mtime +30 -delete 2>/dev/null
else
    # Clear alert file if space is OK
    rm -f "$ALERT_FILE"
fi

# Check database size
if [ -f "$DB_PATH" ]; then
    DB_SIZE_MB=$(du -m "$DB_PATH" | cut -f1)
    if [ "$DB_SIZE_MB" -gt 500 ]; then
        echo "$(date): WARNING - Large database: ${DB_SIZE_MB}MB"
        # Could trigger backup rotation here
    fi
fi
