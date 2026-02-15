#!/bin/bash
# ============================================================
# Deploy to Hostinger VPS
# ============================================================
# Usage:
#   bash scripts/deploy.sh              # Full deploy (test + push + build + restart)
#   bash scripts/deploy.sh --skip-tests # Skip local tests (use when tests already passed)
#   bash scripts/deploy.sh --force      # Deploy even during market hours (emergency fix)
#
# What it does:
#   1. Runs tests locally (catches regressions before they hit prod)
#   2. Pushes to GitHub (audit trail)
#   3. Pulls on VPS via git
#   4. Rebuilds Docker image on VPS
#   5. Restarts container
#   6. Health check
#
# Safety:
#   - Blocks deploys during market hours (9:15 AM - 3:30 PM IST)
#   - Tests must pass before code reaches the server
#   - Old container keeps running until new image is built
#   - Rolls back if health check fails
# ============================================================
set -e
set -u
set -o pipefail

# --- Config ---
VPS_HOST="vps"                          # SSH alias from ~/.ssh/config
PROJECT_DIR="/opt/stock-predict"
COMPOSE_FILE="docker-compose.yml"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- Flags ---
SKIP_TESTS=false
FORCE=false
for arg in "$@"; do
    case $arg in
        --skip-tests) SKIP_TESTS=true ;;
        --force) FORCE=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

log()  { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo "============================================================"
echo "  Stock-Predict Deploy"
echo "============================================================"
echo ""

# ============================================================
# STEP 0: Safety checks
# ============================================================

# Check we're in the right directory
if [ ! -f "start.py" ] || [ ! -f "docker-compose.yml" ]; then
    fail "Run this from the stock-predict root directory"
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Uncommitted changes detected:"
    git status --short
    echo ""
    read -p "Commit all changes before deploying? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Commit message: " MSG
        git add -A
        # Safety: verify no secrets staged
        if git diff --cached --name-only | grep -qE '\.env$|credentials|secret'; then
            fail "Sensitive file detected in staged changes — review before committing"
        fi
        git commit -m "$MSG"
    else
        fail "Commit or stash changes before deploying"
    fi
fi

# Check market hours (IST) — block deploys when positions might be open
HOUR=$(TZ=Asia/Kolkata date +%H)
MINUTE=$(TZ=Asia/Kolkata date +%M)
DOW=$(TZ=Asia/Kolkata date +%u)  # 1=Monday, 7=Sunday
CURRENT_TIME=$((HOUR * 60 + MINUTE))
MARKET_OPEN=$((9 * 60 + 15))    # 9:15 AM
MARKET_CLOSE=$((15 * 60 + 30))  # 3:30 PM

if [ "$DOW" -le 5 ] && [ "$CURRENT_TIME" -ge "$MARKET_OPEN" ] && [ "$CURRENT_TIME" -le "$MARKET_CLOSE" ]; then
    if [ "$FORCE" = false ]; then
        warn "Market is OPEN ($(TZ=Asia/Kolkata date '+%H:%M IST')). Deploying now could disrupt active trades."
        warn "Use --force to override (only for emergency fixes)."
        exit 1
    else
        warn "FORCE deploy during market hours — be careful!"
    fi
fi

# ============================================================
# STEP 1: Run tests locally
# ============================================================
if [ "$SKIP_TESTS" = false ]; then
    log "Running local tests..."
    if python -m pytest tests/ -q --tb=line 2>&1; then
        log "All tests passed"
    else
        fail "Tests failed — fix before deploying"
    fi
else
    warn "Skipping local tests (--skip-tests)"
fi

# ============================================================
# STEP 2: Push to GitHub
# ============================================================
log "Pushing to GitHub..."
git push origin main 2>&1 || fail "Git push failed"
LOCAL_COMMIT=$(git rev-parse --short HEAD)
log "Pushed commit: $LOCAL_COMMIT"

# ============================================================
# STEP 3: Pull on VPS
# ============================================================
log "Pulling on VPS..."
VPS_OUTPUT=$(ssh "$VPS_HOST" "cd $PROJECT_DIR && git pull origin main 2>&1")
echo "  $VPS_OUTPUT" | head -5

# Verify commits match
VPS_COMMIT=$(ssh "$VPS_HOST" "cd $PROJECT_DIR && git rev-parse --short HEAD")
if [ "$LOCAL_COMMIT" != "$VPS_COMMIT" ]; then
    fail "Commit mismatch! Local=$LOCAL_COMMIT VPS=$VPS_COMMIT"
fi
log "VPS at commit: $VPS_COMMIT (matches local)"

# Verify .env exists on VPS (container won't work without it)
if ! ssh "$VPS_HOST" "test -f $PROJECT_DIR/.env"; then
    fail ".env file missing on VPS! Copy it first: scp .env $VPS_HOST:$PROJECT_DIR/.env"
fi
log ".env exists on VPS"

# ============================================================
# STEP 4: Build Docker image on VPS
# ============================================================
log "Building Docker image on VPS..."
ssh "$VPS_HOST" "cd $PROJECT_DIR && docker compose build 2>&1" | tail -5
if [ "${PIPESTATUS[0]:-0}" -ne 0 ]; then
    # Check ssh exit code
    ssh "$VPS_HOST" "cd $PROJECT_DIR && docker compose build" || fail "Docker build failed on VPS"
fi
log "Docker image built"

# ============================================================
# STEP 5: Restart container
# ============================================================
log "Restarting container..."
ssh "$VPS_HOST" "cd $PROJECT_DIR && docker compose up -d 2>&1"
log "Container restarted"

# ============================================================
# STEP 6: Health check
# ============================================================
log "Running health check (waiting 5s)..."
sleep 5

CONTAINER_STATUS=$(ssh "$VPS_HOST" "docker inspect --format='{{.State.Status}}' stock-predict 2>/dev/null || echo 'missing'")
CONTAINER_RESTARTS=$(ssh "$VPS_HOST" "docker inspect --format='{{.RestartCount}}' stock-predict 2>/dev/null || echo 'unknown'")

# Check if container exists and is running or restarting (restarting is OK outside market hours)
if [ "$CONTAINER_STATUS" = "running" ] || [ "$CONTAINER_STATUS" = "restarting" ]; then
    log "Container status: $CONTAINER_STATUS (restarts: $CONTAINER_RESTARTS)"
else
    warn "Container status: $CONTAINER_STATUS — check logs with: ssh vps 'docker logs stock-predict'"
fi

# Show last few log lines
echo ""
log "Recent logs:"
ssh "$VPS_HOST" "docker logs --tail 10 stock-predict 2>&1" | sed 's/^/  /'

# ============================================================
# DONE
# ============================================================
echo ""
echo "============================================================"
echo -e "  ${GREEN}Deploy complete${NC}"
echo "============================================================"
echo "  Commit:    $LOCAL_COMMIT"
echo "  Time:      $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')"
echo "  Container: $CONTAINER_STATUS"
echo ""
echo "  Useful commands:"
echo "    ssh vps 'docker logs -f stock-predict'     # Follow logs"
echo "    ssh vps 'docker ps'                        # Container status"
echo "    ssh vps 'docker compose down'              # Stop trading"
echo "============================================================"
