#!/bin/bash
# Setup systemd service for Linux VPS (Hostinger, DigitalOcean, etc.)
#
# Usage:
#   bash scripts/setup_systemd.sh                    # Default /opt/stock-predict
#   PROJECT_DIR=/home/deploy/app bash scripts/setup_systemd.sh  # Custom path
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
SERVICE_NAME="stock-predict"
TRADING_USER="trading"

echo ""
echo "============================================================"
echo "  Stock-Predict Systemd Service Setup"
echo "============================================================"
echo "  Project:  $PROJECT_DIR"
echo "  Service:  $SERVICE_NAME"
echo "  User:     $TRADING_USER"
echo "============================================================"
echo ""

# Validate project directory
if [ ! -f "$PROJECT_DIR/start.py" ]; then
    echo "ERROR: $PROJECT_DIR/start.py not found."
    echo "Set PROJECT_DIR to the stock-predict root directory."
    exit 1
fi

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "WARNING: $PROJECT_DIR/.env not found. Service will fail without it."
    echo "Copy .env.example to .env and fill in your credentials."
fi

# Create trading user if not exists
if ! id "$TRADING_USER" &>/dev/null; then
    echo "Creating user '$TRADING_USER'..."
    sudo useradd --system --no-create-home --shell /usr/sbin/nologin "$TRADING_USER"
fi

# Create required directories
sudo mkdir -p "$PROJECT_DIR/data" "$PROJECT_DIR/logs"

# Set file permissions
sudo chown -R "$TRADING_USER:$TRADING_USER" "$PROJECT_DIR/data" "$PROJECT_DIR/logs"

if [ -f "$PROJECT_DIR/.env" ]; then
    sudo chown root:"$TRADING_USER" "$PROJECT_DIR/.env"
    sudo chmod 640 "$PROJECT_DIR/.env"
    echo ".env: 640 (root:trading, group-readable)"
fi

sudo chmod 700 "$PROJECT_DIR/data"
sudo chmod 755 "$PROJECT_DIR/logs"

# Create virtual environment if not exists
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/.venv"
    "$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
fi

# Copy and customize service file
echo "Installing systemd service..."
sudo cp "$SCRIPT_DIR/stock-predict.service" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo sed -i "s|/opt/stock-predict|$PROJECT_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo ""
echo "============================================================"
echo "  Service installed!"
echo "============================================================"
echo ""
echo "  Commands:"
echo "    sudo systemctl start $SERVICE_NAME     # Start"
echo "    sudo systemctl stop $SERVICE_NAME      # Stop"
echo "    sudo systemctl status $SERVICE_NAME    # Status"
echo "    journalctl -u $SERVICE_NAME -f         # Follow logs"
echo ""
echo "  For live trading, add to .env:"
echo "    TRADING_MODE=live"
echo "    BROKER=zerodha"
echo ""
echo "  NOTE: Docker is recommended instead. See docker-compose.yml"
echo "============================================================"
