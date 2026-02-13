#!/bin/bash
# macOS LaunchAgent setup for the trading daemon.
#
# Usage:
#   bash scripts/setup_macos.sh install    Install and start the daemon
#   bash scripts/setup_macos.sh uninstall  Stop and remove the daemon
#   bash scripts/setup_macos.sh start      Start the daemon
#   bash scripts/setup_macos.sh stop       Stop the daemon
#   bash scripts/setup_macos.sh status     Show daemon status and recent logs
#   bash scripts/setup_macos.sh logs       Tail today's daemon log

set -e

LABEL="com.trading.daemon"
PROJECT_DIR="/Users/priyeshtiwari/codebase/stock-predict"
PLIST_SRC="$PROJECT_DIR/scripts/com.trading.daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$PROJECT_DIR/logs"

cmd_install() {
    echo "Installing trading daemon LaunchAgent..."

    # Ensure directories exist
    mkdir -p "$HOME/Library/LaunchAgents"
    mkdir -p "$LOG_DIR"
    mkdir -p "$PROJECT_DIR/data/reports"

    # Stop if already loaded
    if launchctl list 2>/dev/null | grep -q "$LABEL"; then
        echo "  Stopping existing daemon..."
        launchctl unload "$PLIST_DST" 2>/dev/null || true
    fi

    # Copy plist
    cp "$PLIST_SRC" "$PLIST_DST"
    echo "  Plist installed to $PLIST_DST"

    # Load and start
    launchctl load "$PLIST_DST"
    echo "  Daemon loaded and started"

    echo ""
    echo "Done! The trading daemon will now:"
    echo "  - Start automatically when you log in"
    echo "  - Run paper trading during market hours (9:15-15:30 IST)"
    echo "  - Sleep during non-market hours (low CPU)"
    echo "  - Restart automatically if it crashes"
    echo ""
    echo "Commands:"
    echo "  bash scripts/setup_macos.sh status   Check if running"
    echo "  bash scripts/setup_macos.sh logs     Watch live logs"
    echo "  python trade.py report               View daily P&L report"
    echo "  bash scripts/setup_macos.sh stop     Stop the daemon"
}

cmd_uninstall() {
    echo "Uninstalling trading daemon LaunchAgent..."

    if [ -f "$PLIST_DST" ]; then
        launchctl unload "$PLIST_DST" 2>/dev/null || true
        rm "$PLIST_DST"
        echo "  Daemon stopped and plist removed"
    else
        echo "  LaunchAgent not installed"
    fi

    echo "Done."
}

cmd_start() {
    if ! [ -f "$PLIST_DST" ]; then
        echo "Daemon not installed. Run: bash scripts/setup_macos.sh install"
        exit 1
    fi

    launchctl start "$LABEL"
    echo "Daemon started"
}

cmd_stop() {
    launchctl stop "$LABEL" 2>/dev/null || true
    echo "Daemon stopped"
}

cmd_status() {
    echo "=== Trading Daemon Status ==="
    echo ""

    # Check launchctl
    ENTRY=$(launchctl list 2>/dev/null | grep "$LABEL" || true)
    if [ -n "$ENTRY" ]; then
        PID=$(echo "$ENTRY" | awk '{print $1}')
        EXIT_CODE=$(echo "$ENTRY" | awk '{print $2}')
        if [ "$PID" != "-" ] && [ -n "$PID" ]; then
            echo "Status:    RUNNING (PID: $PID)"
        else
            echo "Status:    LOADED (not currently running, last exit: $EXIT_CODE)"
        fi
    else
        echo "Status:    NOT INSTALLED"
        echo ""
        echo "Install with: bash scripts/setup_macos.sh install"
        return
    fi

    # Show today's log
    TODAY=$(date +%Y-%m-%d)
    LOG_FILE="$LOG_DIR/daemon_$TODAY.log"
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== Recent Log (last 15 lines) ==="
        tail -15 "$LOG_FILE"
    fi

    # Show latest report
    REPORT="$PROJECT_DIR/data/reports/$TODAY.txt"
    if [ -f "$REPORT" ]; then
        echo ""
        echo "=== Today's Report ==="
        cat "$REPORT"
    fi

    # Show issues count
    ISSUES_FILE="$LOG_DIR/issues_$TODAY.log"
    if [ -f "$ISSUES_FILE" ]; then
        ISSUE_COUNT=$(wc -l < "$ISSUES_FILE" | tr -d ' ')
        if [ "$ISSUE_COUNT" -gt 0 ]; then
            echo ""
            echo "=== Issues: $ISSUE_COUNT warnings/errors ==="
            echo "Run: bash scripts/setup_macos.sh issues"
        fi
    fi

    echo ""
    echo "Log file: $LOG_FILE"
}

cmd_issues() {
    TODAY=$(date +%Y-%m-%d)
    ISSUES_FILE="$LOG_DIR/issues_$TODAY.log"

    if [ ! -f "$ISSUES_FILE" ] || [ ! -s "$ISSUES_FILE" ]; then
        echo "No issues today!"
        return
    fi

    TOTAL=$(wc -l < "$ISSUES_FILE" | tr -d ' ')
    echo "=== Issues Report ($TODAY) - $TOTAL total ==="
    echo ""

    # Count by category (grep -c returns 0 with exit code 1 on no match, so use || true)
    RATE_LIMIT=$(grep -c -i -E '403|rate.limit|forbidden' "$ISSUES_FILE" 2>/dev/null || true)
    DATA_GAPS=$(grep -c -i -E 'no data|fetch.*fail|timeout|yfinance' "$ISSUES_FILE" 2>/dev/null || true)
    BROKER=$(grep -c -i -E 'broker|order|token' "$ISSUES_FILE" 2>/dev/null || true)
    RATE_LIMIT=${RATE_LIMIT:-0}
    DATA_GAPS=${DATA_GAPS:-0}
    BROKER=${BROKER:-0}

    CLASSIFIED=$((RATE_LIMIT + DATA_GAPS + BROKER))
    OTHER=$((TOTAL - CLASSIFIED))
    if [ "$OTHER" -lt 0 ]; then OTHER=0; fi

    echo "  API Rate Limits:  $RATE_LIMIT"
    echo "  Data Gaps:        $DATA_GAPS"
    echo "  Broker/Orders:    $BROKER"
    echo "  Other:            $OTHER"
    echo ""

    echo "--- Unique issues (deduplicated) ---"
    echo ""

    # Show unique issue messages (last field after |, deduplicated)
    awk -F'|' '{print $NF}' "$ISSUES_FILE" | sort -u | head -30

    echo ""
    echo "Full log: $ISSUES_FILE"
}

cmd_logs() {
    TODAY=$(date +%Y-%m-%d)
    LOG_FILE="$LOG_DIR/daemon_$TODAY.log"

    if [ -f "$LOG_FILE" ]; then
        echo "Tailing $LOG_FILE (Ctrl+C to stop)..."
        echo ""
        tail -f "$LOG_FILE"
    else
        echo "No log file for today: $LOG_FILE"
        echo "Is the daemon running? Check: bash scripts/setup_macos.sh status"
    fi
}

# Route command
case "${1:-}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    status)    cmd_status ;;
    issues)    cmd_issues ;;
    logs)      cmd_logs ;;
    *)
        echo "Usage: bash scripts/setup_macos.sh {install|uninstall|start|stop|status|issues|logs}"
        echo ""
        echo "Commands:"
        echo "  install    Install LaunchAgent and start daemon"
        echo "  uninstall  Stop daemon and remove LaunchAgent"
        echo "  start      Start the daemon"
        echo "  stop       Stop the daemon"
        echo "  status     Show daemon status and recent logs"
        echo "  issues     Show today's issues (errors/warnings, categorized)"
        echo "  logs       Tail today's daemon log (live)"
        exit 1
        ;;
esac
