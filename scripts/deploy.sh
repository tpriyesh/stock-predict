#!/bin/bash
# ============================================================
# Deploy stock-predict to Hostinger VPS
# ============================================================
# Usage:
#   bash scripts/deploy.sh deploy@your-vps-ip
#   SSH_PORT=2222 bash scripts/deploy.sh deploy@your-vps-ip
#
# Prerequisites:
#   1. VPS hardened with: bash scripts/secure_vps.sh
#   2. SSH key auth configured
#   3. .env file copied to server separately (never synced!)
#
# What it does:
#   1. rsync code to /opt/stock-predict (excludes .env, data, .git)
#   2. Rebuild and restart Docker containers
# ============================================================
set -euo pipefail

SSH_PORT="${SSH_PORT:-2222}"
REMOTE_DIR="${REMOTE_DIR:-/opt/stock-predict}"

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/deploy.sh [user@host]"
    echo ""
    echo "Examples:"
    echo "  bash scripts/deploy.sh deploy@192.168.1.100"
    echo "  SSH_PORT=2222 bash scripts/deploy.sh deploy@my-vps.com"
    echo ""
    echo "First time? Copy .env to the server manually:"
    echo "  scp -P $SSH_PORT .env deploy@your-vps:$REMOTE_DIR/.env"
    exit 1
fi

TARGET="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "============================================================"
echo "  Deploying stock-predict"
echo "============================================================"
echo "  Target:  $TARGET"
echo "  Port:    $SSH_PORT"
echo "  Remote:  $REMOTE_DIR"
echo "============================================================"
echo ""

# Step 1: Sync code (exclude secrets, data, and dev files)
echo "[1/3] Syncing code..."
rsync -avz --delete \
    -e "ssh -p $SSH_PORT" \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='data/' \
    --exclude='logs/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    --exclude='node_modules/' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    "$PROJECT_DIR/" "$TARGET:$REMOTE_DIR/"

# Step 2: Check .env exists on remote
echo ""
echo "[2/3] Checking remote .env..."
# shellcheck disable=SC2029
if ! ssh -p "$SSH_PORT" "$TARGET" "test -f $REMOTE_DIR/.env"; then
    echo "WARNING: No .env file found on server!"
    echo "Copy it manually: scp -P $SSH_PORT .env $TARGET:$REMOTE_DIR/.env"
    echo "Then run this script again."
    exit 1
fi

# Step 3: Rebuild and restart
echo ""
echo "[3/3] Rebuilding and restarting containers..."
# shellcheck disable=SC2029
ssh -p "$SSH_PORT" "$TARGET" "cd $REMOTE_DIR && docker compose build --no-cache && docker compose up -d"

echo ""
echo "============================================================"
echo "  Deployed successfully!"
echo "============================================================"
echo ""
echo "  Check status:  ssh -p $SSH_PORT $TARGET 'docker logs -f stock-predict'"
echo "  Live mode:     ssh -p $SSH_PORT $TARGET 'cd $REMOTE_DIR && docker compose -f docker-compose.yml -f docker-compose.live.yml up -d'"
echo "============================================================"
