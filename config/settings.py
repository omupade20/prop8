# config/settings.py
import os

"""
Settings for connecting with the Upstox API.
Paste your daily access token below whenever you generate it
from the Upstox app or Developer Console.
"""

# Paste your daily Upstox token between the quotes:
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzVkNVWTMiLCJqdGkiOiI2YWQ5Yjc4OWIxM2NkZDdlMjBjOTZmNjciLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzkyNjUzMTkzLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3OTI3MDY0MDB9.MisuFPt8DSxxLM6Gf_CU8X_SliXov-gFevEYwvF5VAQ"

# Cloud fallback: If deployed on Render/Cloud, read from environment variable
if not ACCESS_TOKEN:
    ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "").strip().strip('"').strip("'")

# Base REST API URL (v2)
API_BASE_URL = "https://api.upstox.com/v2/"
