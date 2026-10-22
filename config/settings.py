import os

# Priority: Environment variable (Render cloud deployment / .env) -> Fallback
ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip().strip('"').strip("'")

# Base REST API URL (v2)
API_BASE_URL = os.environ.get("UPSTOX_API_BASE_URL", "https://api.upstox.com/v2/")

