#!/usr/bin/env python3
"""
Upstox OAuth Helper - Get access token for API access.

Upstox uses OAuth2, so you need to:
1. Create an app at https://developer.upstox.com/
2. Get API Key and Secret
3. Run this script to get Access Token
4. Add token to .env file

Usage:
    python scripts/upstox_auth.py
"""
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
from pathlib import Path
from dotenv import load_dotenv, set_key

# Load existing env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback."""

    def do_GET(self):
        """Process OAuth callback."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if 'code' in params:
            self.server.auth_code = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                    <h1>Authentication Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """)
        else:
            self.server.auth_code = None
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                    <h1>Authentication Failed</h1>
                    <p>No authorization code received.</p>
                </body>
                </html>
            """)

    def log_message(self, format, *args):
        """Suppress logging."""
        pass


def get_access_token():
    """Interactive OAuth flow to get access token."""
    print("=" * 60)
    print("Upstox OAuth Authentication")
    print("=" * 60)
    print()

    # Get credentials
    api_key = os.getenv("UPSTOX_API_KEY", "")
    api_secret = os.getenv("UPSTOX_API_SECRET", "")

    if not api_key:
        print("Enter your Upstox API Key")
        print("(Get it from https://developer.upstox.com/)")
        api_key = input("API Key: ").strip()

    if not api_secret:
        print("\nEnter your Upstox API Secret")
        api_secret = input("API Secret: ").strip()

    if not api_key or not api_secret:
        print("\n❌ API Key and Secret are required!")
        return None

    # OAuth configuration
    redirect_uri = "http://127.0.0.1:5000/callback"
    auth_url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?client_id={api_key}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
    )

    print("\n📱 Opening browser for Upstox login...")
    print("   If browser doesn't open, visit this URL:")
    print(f"   {auth_url}")
    print()

    # Start local server for callback
    server = HTTPServer(('127.0.0.1', 5000), OAuthCallbackHandler)
    server.auth_code = None

    # Open browser
    webbrowser.open(auth_url)

    print("⏳ Waiting for authentication...")
    print("   (Login to Upstox in your browser)")
    print()

    # Wait for callback (with timeout)
    server.timeout = 300  # 5 minutes
    while server.auth_code is None:
        server.handle_request()

    auth_code = server.auth_code
    server.server_close()

    if not auth_code:
        print("❌ Failed to get authorization code")
        return None

    print("✅ Got authorization code!")
    print("📡 Exchanging for access token...")

    # Exchange code for token
    token_url = "https://api.upstox.com/v2/login/authorization/token"
    response = requests.post(
        token_url,
        data={
            "code": auth_code,
            "client_id": api_key,
            "client_secret": api_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code"
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.status_code != 200:
        print(f"❌ Token exchange failed: {response.text}")
        return None

    data = response.json()
    access_token = data.get("access_token")

    if not access_token:
        print(f"❌ No access token in response: {data}")
        return None

    print("✅ Got access token!")
    print()

    # Save to .env
    print("💾 Saving credentials to .env...")

    # Create or update .env file
    if not env_path.exists():
        env_path.touch()

    set_key(str(env_path), "UPSTOX_API_KEY", api_key)
    set_key(str(env_path), "UPSTOX_API_SECRET", api_secret)
    set_key(str(env_path), "UPSTOX_ACCESS_TOKEN", access_token)

    print("✅ Credentials saved!")
    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("You can now run:")
    print("  python trade.py test       # Test connection")
    print("  python trade.py start      # Start paper trading")
    print("  python daemon.py run       # Run daemon")
    print()
    print("⚠️  NOTE: Upstox tokens expire daily!")
    print("   Run this script again if you get auth errors.")
    print()

    return access_token


def test_connection(access_token: str) -> bool:
    """Test if access token is valid."""
    response = requests.get(
        "https://api.upstox.com/v2/user/profile",
        headers={"Authorization": f"Bearer {access_token}"}
    )

    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "success":
            user = data.get("data", {})
            print(f"✅ Connected as: {user.get('user_name', 'Unknown')}")
            return True

    return False


if __name__ == "__main__":
    # Check if token already exists and is valid
    existing_token = os.getenv("UPSTOX_ACCESS_TOKEN")

    if existing_token:
        print("Found existing token in .env")
        print("Testing connection...")

        if test_connection(existing_token):
            print("\n✅ Existing token is valid!")
            print("   No re-authentication needed.")

            response = input("\nRe-authenticate anyway? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(0)
        else:
            print("⚠️  Existing token is invalid or expired")
            print()

    get_access_token()
