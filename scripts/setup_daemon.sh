#!/bin/bash
# Setup Trading Daemon for macOS
# This script installs the daemon to run automatically on login

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.trading.daemon.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "========================================"
echo "Trading Daemon Setup"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "⚠️  No .env file found!"
    echo "   Creating from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo ""
    echo "   IMPORTANT: Edit .env and add your API keys:"
    echo "   - UPSTOX_API_KEY"
    echo "   - UPSTOX_API_SECRET"
    echo "   - UPSTOX_ACCESS_TOKEN"
    echo ""
fi

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Update plist with correct paths
echo "📝 Configuring daemon..."
PYTHON_PATH=$(which python3)
sed -i '' "s|/usr/bin/python3|$PYTHON_PATH|g" "$PLIST_SRC" 2>/dev/null || true
sed -i '' "s|/Users/priyeshtiwari/codebase/stock-predict|$PROJECT_DIR|g" "$PLIST_SRC" 2>/dev/null || true

# Copy plist to LaunchAgents
echo "📦 Installing daemon..."
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DEST"

# Unload if already loaded
launchctl unload "$PLIST_DEST" 2>/dev/null || true

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Available commands:"
echo ""
echo "  Start daemon:    launchctl load $PLIST_DEST"
echo "  Stop daemon:     launchctl unload $PLIST_DEST"
echo "  Check status:    python3 $PROJECT_DIR/daemon.py status"
echo ""
echo "Or run manually:"
echo ""
echo "  python3 $PROJECT_DIR/daemon.py run    # Foreground (for testing)"
echo "  python3 $PROJECT_DIR/daemon.py start  # Background"
echo "  python3 $PROJECT_DIR/daemon.py stop   # Stop"
echo ""
echo "The daemon will automatically:"
echo "  ✓ Start when you log in"
echo "  ✓ Sleep when market is closed (low CPU)"
echo "  ✓ Wake up before market open"
echo "  ✓ Run trading session during market hours"
echo "  ✓ Generate reports after market close"
echo ""

# Ask to start now
read -p "Start daemon now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    launchctl load "$PLIST_DEST"
    echo "✅ Daemon started!"
    sleep 2
    python3 "$PROJECT_DIR/daemon.py" status
fi
