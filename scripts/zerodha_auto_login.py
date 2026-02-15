#!/usr/bin/env python3
"""
Zerodha Kite Connect - Automated Daily Token Refresh.

Performs the full login flow using TOTP (no browser needed):
1. POST credentials to get request_id
2. POST TOTP for 2FA
3. GET request_token from OAuth redirect
4. Generate access_token via Kite SDK

Usage:
    python scripts/zerodha_auto_login.py

Cron (run daily at 7:30 AM on weekdays):
    30 7 * * 1-5 cd /path/to/stock-predict && python scripts/zerodha_auto_login.py

All credentials read from .env file.
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import requests
import pyotp
from kiteconnect import KiteConnect

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# URLs
LOGIN_URL = "https://kite.zerodha.com/api/login"
TWOFA_URL = "https://kite.zerodha.com/api/twofa"
CONNECT_URL = "https://kite.trade/connect/login"

# Token save path
TOKEN_PATH = Path(__file__).parent.parent / "data" / "zerodha_token.json"

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 5


def _send_telegram(message: str):
    """Send notification via Telegram (best-effort)."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=10
        )
    except Exception:
        pass


def _save_token(access_token: str):
    """Save token to file and update .env."""
    # Save to JSON file
    data = {
        "access_token": access_token,
        "timestamp": datetime.now().isoformat(),
    }
    TOKEN_PATH.write_text(json.dumps(data, indent=2))
    print(f"  Token saved to {TOKEN_PATH}")

    # Update .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        content = env_path.read_text()
        lines = content.split("\n")
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("ZERODHA_ACCESS_TOKEN="):
                lines[i] = f"ZERODHA_ACCESS_TOKEN={access_token}"
                updated = True
                break
        if not updated:
            lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}")
        env_path.write_text("\n".join(lines))
        print(f"  .env updated")


def perform_login() -> str:
    """
    Perform the full Zerodha login flow.

    Returns:
        access_token on success

    Raises:
        Exception on failure
    """
    user_id = os.getenv("ZERODHA_USER_ID", "")
    password = os.getenv("ZERODHA_PASSWORD", "")
    totp_key = os.getenv("ZERODHA_TOTP_KEY", "")
    api_key = os.getenv("ZERODHA_API_KEY", "")
    api_secret = os.getenv("ZERODHA_API_SECRET", "")

    # Validate all credentials present
    missing = []
    if not user_id:
        missing.append("ZERODHA_USER_ID")
    if not password:
        missing.append("ZERODHA_PASSWORD")
    if not totp_key:
        missing.append("ZERODHA_TOTP_KEY")
    if not api_key:
        missing.append("ZERODHA_API_KEY")
    if not api_secret:
        missing.append("ZERODHA_API_SECRET")

    if missing:
        raise ValueError(f"Missing credentials in .env: {', '.join(missing)}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })

    # Step 1: Login with credentials
    print("  Step 1: Logging in...")
    resp = session.post(
        LOGIN_URL,
        data={"user_id": user_id, "password": password},
        timeout=30
    )
    if resp.status_code != 200:
        raise Exception(f"Login failed: HTTP {resp.status_code}")

    login_data = resp.json()
    if login_data.get("status") != "success":
        raise Exception(f"Login failed: {login_data.get('message', 'Unknown error')}")

    request_id = login_data["data"]["request_id"]

    # Step 2: TOTP 2FA
    print("  Step 2: TOTP 2FA...")
    totp_code = pyotp.TOTP(totp_key).now()

    resp = session.post(
        TWOFA_URL,
        data={
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp_code,
        },
        timeout=30
    )
    if resp.status_code != 200:
        raise Exception(f"2FA failed: HTTP {resp.status_code}")

    # Step 3: Get request token from OAuth redirect
    # Follow redirects manually — stop before hitting localhost (which won't be running)
    print("  Step 3: Getting request token...")
    redirect_url = f"{CONNECT_URL}?api_key={api_key}&v=3"

    for _ in range(10):  # max 10 hops
        resp = session.get(redirect_url, timeout=30, allow_redirects=False)
        if resp.status_code not in (301, 302):
            break
        redirect_url = resp.headers.get("Location", "")
        if "request_token=" in redirect_url:
            break  # Found it — don't follow further (next hop is localhost)

    if "request_token=" not in redirect_url:
        raise Exception(f"No request_token in redirect URL: {redirect_url}")

    request_token = redirect_url.split("request_token=")[1].split("&")[0]

    # Step 4: Generate access token
    print("  Step 4: Generating access token...")
    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]

    # Verify token works
    kite.set_access_token(access_token)
    profile = kite.profile()
    user_name = profile.get("user_name", "Unknown")
    print(f"  Verified: logged in as {user_name}")

    return access_token


def auto_login() -> bool:
    """
    Perform login with retries.

    Returns:
        True on success, False on failure.
    """
    print(f"\nZerodha Auto-Login - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\nAttempt {attempt}/{MAX_RETRIES}:")
            access_token = perform_login()

            _save_token(access_token)
            _send_telegram(f"Zerodha token refreshed successfully at {datetime.now().strftime('%H:%M')}")

            print(f"\nLogin successful!")
            print(f"Token: {access_token[:10]}...{access_token[-5:]}")
            return True

        except Exception as e:
            print(f"\n  Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

    # All attempts failed
    msg = f"Zerodha auto-login FAILED after {MAX_RETRIES} attempts"
    print(f"\n{msg}")
    _send_telegram(f"CRITICAL: {msg}. Manual login needed!")
    return False


if __name__ == "__main__":
    success = auto_login()
    sys.exit(0 if success else 1)
