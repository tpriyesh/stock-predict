#!/bin/bash
# Wrapper script for launchd to run the trading daemon.
# Loads .env, sets paper mode, and execs the daemon in foreground.
# launchd manages the process lifecycle (restart on crash, etc.)

set -e

PROJECT_DIR="/Users/priyeshtiwari/codebase/stock-predict"
PYTHON="/Users/priyeshtiwari/.pyenv/versions/3.11.5/bin/python3"

cd "$PROJECT_DIR"

# Load environment variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Force paper trading mode (safety: never default to live from daemon)
export TRADING_MODE=paper

# Ensure logs and data directories exist
mkdir -p logs data data/reports

# Run daemon in foreground (launchd manages the process)
exec "$PYTHON" daemon.py run
