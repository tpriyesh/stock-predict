# ============================================
# Stock-Predict Trading Agent
# ============================================
# IST timezone for Indian market trading
# Usage:
#   docker build -t stock-predict .
#   docker run --env-file .env stock-predict              # Paper mode
#   docker run --env-file .env stock-predict --live        # Live mode
#   docker run --env-file .env stock-predict --zerodha     # Zerodha
#   docker run --rm stock-predict pytest tests/ -v         # Run tests

FROM python:3.11-slim-bookworm

# System deps + IST timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl \
    && ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-root user (nologin shell — no interactive access needed)
RUN useradd --no-log-init --create-home --shell /usr/sbin/nologin trading

WORKDIR /app

# Install Python deps (cached layer — lean set, no UI/streamlit)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && pip install --no-cache-dir pytest

# Copy application code
COPY --chown=trading:trading . .

# Create required directories
RUN mkdir -p /app/data /app/data/reports /app/logs \
    && chown -R trading:trading /app/data /app/logs

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Kolkata \
    TRADING_MODE=paper

USER trading

# Health check — verify config loads and providers initialize
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from config.trading_config import CONFIG; from providers.quota import get_quota_manager; print('OK')" || exit 1

ENTRYPOINT ["python", "start.py"]
