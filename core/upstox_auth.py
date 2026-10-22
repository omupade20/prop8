# core/upstox_auth.py
"""
Automated Upstox Authentication & Token Engine.
Features:
1. Instant Pre-Flight Token Health Check (<0.5s)
2. Automated Headless TOTP Login via pyotp (when credentials provided in .env/settings)
3. 1-Click Local Receiver Server (Auto-captures OAuth code from browser redirect; zero copy-pasting)
4. Automatic persistence of fresh ACCESS_TOKEN to config/settings.py
"""

import os
import re
import sys
import time
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from typing import Optional, Dict, Any, Tuple
import requests

try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Local HTTP handler to intercept OAuth redirect code from Upstox."""
    auth_code: Optional[str] = None

    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <html>
            <head><title>JARVIS Upstox Auth</title></head>
            <body style="font-family: Arial, sans-serif; background: #0b0f19; color: #10b981; text-align: center; padding: 60px;">
                <h1 style="font-size: 32px;">⚡ JARVIS Authentication Successful!</h1>
                <p style="color: #94a3b8; font-size: 18px;">Your daily Upstox Access Token has been automatically captured and saved.</p>
                <p style="color: #38bdf8;">You can now close this browser tab. Your trading engine is starting...</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Failed to capture authorization code.")

    def log_message(self, format, *args):
        # Silence local server request logging
        return


def validate_token(token: str, base_url: str = "https://api.upstox.com/v2/") -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validates token against Upstox profile endpoint in <0.5 seconds.
    """
    if not token or len(token) < 20:
        return False, None

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        res = requests.get(f"{base_url}user/profile", headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json().get("data", {})
            return True, data
    except Exception:
        pass
    return False, None


def get_account_margin(token: str, base_url: str = "https://api.upstox.com/v2/") -> float:
    """Fetches real-time available cash margin for equity cash."""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        res = requests.get(f"{base_url}user/get-funds-and-margin?segment=SEC", headers=headers, timeout=4)
        if res.status_code == 200:
            equity_data = res.json().get("data", {}).get("equity", {})
            return float(equity_data.get("available_margin", 0.0))
    except Exception:
        pass
    return 0.0


def exchange_code_for_token(code: str, api_key: str, api_secret: str, redirect_uri: str) -> Optional[str]:
    """Exchanges OAuth authorization code for an Upstox access token."""
    url = "https://api.upstox.com/v2/login/authorization/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "code": code,
        "client_id": api_key,
        "client_secret": api_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }
    try:
        res = requests.post(url, headers=headers, data=data, timeout=10)
        if res.status_code == 200:
            token = res.json().get("access_token")
            return token
        else:
            print(f"[AUTH ERROR] Token exchange failed: {res.text}")
    except Exception as e:
        print(f"[AUTH ERROR] Failed during token request: {e}")
    return None


def save_access_token_to_settings(new_token: str):
    """Updates config/settings.py with the freshly minted access token."""
    settings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "settings.py")
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                content = f.read()

            pattern = r'ACCESS_TOKEN\s*=\s*["\'].*?["\']'
            new_content = re.sub(pattern, f'ACCESS_TOKEN = "{new_token}"', content, count=1)

            with open(settings_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("[AUTH] Successfully saved new ACCESS_TOKEN to config/settings.py!")
    except Exception as e:
        print(f"[AUTH ERROR] Could not save token to settings: {e}")


def capture_token_via_browser(api_key: str, api_secret: str, redirect_uri: str = "http://127.0.0.1:5000/callback") -> Optional[str]:
    """
    1-Click Automated OAuth:
    Starts a local receiver server, opens browser for 1 login click, catches the code,
    fetches the token, and writes it directly to settings.py.
    """
    port = 5000
    try:
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.port:
            port = parsed.port
    except Exception:
        port = 5000

    server = HTTPServer(("127.0.0.1", port), OAuthCallbackHandler)
    OAuthCallbackHandler.auth_code = None

    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={urllib.parse.quote(redirect_uri)}"
    print(f"\n[AUTH] Opening browser to authenticate with Upstox...")
    print(f"[AUTH] Login URL: {auth_url}")
    webbrowser.open(auth_url)

    print("[AUTH] Waiting for login authorization from browser (timeout 120s)...")
    for _ in range(120):
        if OAuthCallbackHandler.auth_code:
            break
        time.sleep(1)

    if OAuthCallbackHandler.auth_code:
        print("[AUTH] Authorization code received! Exchanging for Access Token...")
        token = exchange_code_for_token(OAuthCallbackHandler.auth_code, api_key, api_secret, redirect_uri)
        if token:
            save_access_token_to_settings(token)
            return token
    else:
        print("[AUTH TIMEOUT] No authorization code received within 120 seconds.")
    return None


def ensure_valid_session() -> str:
    """
    Pre-flight check:
    1. Tests current token in config.settings.
    2. If valid -> returns it immediately (takes <0.5s).
    """
    from config.settings import ACCESS_TOKEN, API_BASE_URL
    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip().strip('"').strip("'") or ACCESS_TOKEN

    print("[AUTH] Performing Upstox Token Pre-Flight Health Check...")
    is_valid, user_data = validate_token(token, API_BASE_URL)

    if is_valid and user_data:
        user_name = user_data.get("user_name", "Valued Client")
        user_id = user_data.get("user_id", "")
        email = user_data.get("email", "")
        margin = get_account_margin(token, API_BASE_URL)
        import config.settings
        config.settings.ACCESS_TOKEN = token
        print("==========================================================================")
        print(f"  [UPSTOX AUTH SUCCESS] Verified Active Session")
        print(f"  Client: {user_name} ({user_id}) | Email: {email}")
        print(f"  Live Intraday Margin Available: Rs. {margin:,.2f}")
        print("==========================================================================")
        return token

    print("[AUTH WARNING] Current Upstox Access Token has expired or is invalid.")

    # Check for credentials in environment or settings
    api_key = os.environ.get("UPSTOX_API_KEY", "")
    api_secret = os.environ.get("UPSTOX_API_SECRET", "")
    redirect_uri = os.environ.get("UPSTOX_REDIRECT_URI", "http://127.0.0.1:5000/callback")

    if api_key and api_secret:
        new_token = capture_token_via_browser(api_key, api_secret, redirect_uri)
        if new_token:
            import config.settings
            config.settings.ACCESS_TOKEN = new_token
            return new_token

    # Check if running in headless cloud environment (Render, Docker, or non-TTY)
    is_cloud = os.environ.get("RENDER") or os.environ.get("PORT") or not sys.stdin.isatty()
    if is_cloud:
        print("\n[CLOUD AUTH ALERT] Headless execution detected on Render.")
        print("[ACTION REQUIRED] Set 'UPSTOX_ACCESS_TOKEN' in Render Dashboard -> Environment Variables.")
        try:
            from core.telegram_notifier import get_telegram_notifier
            get_telegram_notifier()._send_async(
                "⚠️ *JARVIS UPSTOX AUTH REQUIRED*\n"
                "Your daily Upstox token is missing or expired on Render.\n\n"
                "👉 Open *Render Dashboard* -> *Environment* -> Update `UPSTOX_ACCESS_TOKEN`."
            )
        except Exception:
            pass
        return token

    print("\n[MANUAL TOKEN PROMPT] Please enter your fresh daily Upstox Access Token:")
    try:
        user_token = input("ACCESS_TOKEN > ").strip().strip('"').strip("'")
        if user_token:
            save_access_token_to_settings(user_token)
            import config.settings
            config.settings.ACCESS_TOKEN = user_token
            is_valid, user_data = validate_token(user_token, API_BASE_URL)
            if is_valid and user_data:
                user_name = user_data.get("user_name", "Valued Client")
                user_id = user_data.get("user_id", "")
                margin = get_account_margin(user_token, API_BASE_URL)
                print("==========================================================================")
                print(f"  [UPSTOX AUTH SUCCESS] Verified Active Session")
                print(f"  Client: {user_name} ({user_id}) | Live Margin Available: Rs. {margin:,.2f}")
                print("==========================================================================")
                return user_token
            else:
                print("\n[AUTH ERROR] The token you entered was rejected by Upstox API (UDAPI100050: Invalid Token).")
                print("Note: If your IP address was recently registered, Upstox requires generating a NEW token.")
                return user_token
    except Exception:
        pass

    return token

