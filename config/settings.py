# config/settings.py
import os

"""
Settings for connecting with the Upstox API.
Paste your daily access token below whenever you generate it
from the Upstox app or Developer Console.
"""

# Paste your daily Upstox token between the quotes:
ACCESS_TOKEN = ""

# Cloud fallback: If deployed on Render/Cloud, read from environment variable
if not ACCESS_TOKEN:
    ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip().strip('"').strip("'")

# Base REST API URL (v2)
API_BASE_URL = "https://api.upstox.com/v2/"
