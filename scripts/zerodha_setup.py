#!/usr/bin/env python3
"""
Zerodha Kite Connect - First Time Setup.

Interactive script that:
1. Collects your Zerodha credentials
2. Tests TOTP generation
3. Performs initial login to verify everything works
4. Saves credentials to .env
5. Prints cron job command for daily auto-login

Run once:
    python scripts/zerodha_setup.py
"""
import os
import sys
from pathlib import Path
from getpass import getpass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ENV_PATH = PROJECT_ROOT / ".env"


def _update_env(key: str, value: str):
    """Add or update a key in .env file."""
    if ENV_PATH.exists():
        content = ENV_PATH.read_text()
        lines = content.split("\n")
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}")
        ENV_PATH.write_text("\n".join(lines))
    else:
        ENV_PATH.write_text(f"{key}={value}\n")


def main():
    print()
    print("=" * 60)
    print("  ZERODHA KITE CONNECT - FIRST TIME SETUP")
    print("=" * 60)
    print()
    print("Before starting, make sure you have:")
    print("  1. A Zerodha trading account")
    print("  2. Kite Connect app created at https://developers.kite.trade")
    print("  3. TOTP enabled in Zerodha settings (Security > 2FA)")
    print("     When enabling, click 'Can't scan?' to get the TOTP secret key")
    print()

    input("Press Enter to continue...")
    print()

    # Collect credentials
    print("Step 1: Enter your credentials")
    print("-" * 40)

    user_id = input("Zerodha User ID (e.g., AB1234): ").strip()
    password = getpass("Zerodha Password: ").strip()
    totp_key = input("TOTP Secret Key (from 2FA setup): ").strip()
    api_key = input("Kite Connect API Key: ").strip()
    api_secret = getpass("Kite Connect API Secret: ").strip()

    if not all([user_id, password, totp_key, api_key, api_secret]):
        print("\nAll fields are required. Please try again.")
        return

    # Test TOTP
    print()
    print("Step 2: Testing TOTP generation")
    print("-" * 40)
    try:
        import pyotp
        totp = pyotp.TOTP(totp_key)
        code = totp.now()
        print(f"  Generated TOTP code: {code}")
        print("  TOTP generation works!")
    except Exception as e:
        print(f"  TOTP generation failed: {e}")
        print("  Check your TOTP secret key and try again.")
        return

    # Save to .env
    print()
    print("Step 3: Saving credentials to .env")
    print("-" * 40)

    _update_env("ZERODHA_USER_ID", user_id)
    _update_env("ZERODHA_PASSWORD", password)
    _update_env("ZERODHA_TOTP_KEY", totp_key)
    _update_env("ZERODHA_API_KEY", api_key)
    _update_env("ZERODHA_API_SECRET", api_secret)
    _update_env("BROKER", "zerodha")
    print(f"  Saved to {ENV_PATH}")

    # Reload env
    os.environ["ZERODHA_USER_ID"] = user_id
    os.environ["ZERODHA_PASSWORD"] = password
    os.environ["ZERODHA_TOTP_KEY"] = totp_key
    os.environ["ZERODHA_API_KEY"] = api_key
    os.environ["ZERODHA_API_SECRET"] = api_secret

    # Test login
    print()
    print("Step 4: Testing full login flow")
    print("-" * 40)
    try:
        from scripts.zerodha_auto_login import perform_login, _save_token
        access_token = perform_login()
        _save_token(access_token)
        print("  Login successful!")
    except Exception as e:
        print(f"  Login failed: {e}")
        print()
        print("  Credentials are saved. You can retry with:")
        print("    python scripts/zerodha_auto_login.py")
        return

    # Done
    print()
    print("=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Your Zerodha integration is ready. Here's what to do next:")
    print()
    print("  Test connection:")
    print("    python trade.py test --zerodha")
    print()
    print("  Start paper trading:")
    print("    python trade.py start")
    print()
    print("  Start live trading with Zerodha:")
    print("    python start.py --zerodha")
    print()
    print("  Set up daily auto-login (add to crontab):")
    print(f"    30 7 * * 1-5 cd {PROJECT_ROOT} && python scripts/zerodha_auto_login.py >> logs/auto_login.log 2>&1")
    print()


if __name__ == "__main__":
    main()
