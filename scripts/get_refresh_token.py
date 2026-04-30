#!/usr/bin/env python3
"""Get YouTube API refresh token via OAuth flow."""

import os
import sys

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube"]


def main():
    load_dotenv()

    # Check required env vars
    client_id = os.environ.get("YT_CLIENT_ID")
    client_secret = os.environ.get("YT_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("Error: YT_CLIENT_ID and YT_CLIENT_SECRET must be set in .env", file=sys.stderr)
        return 1

    # Build client config from env vars
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": os.environ.get("YT_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.environ.get("YT_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

    print()
    print("Opening browser for OAuth consent...")
    print("(If browser doesn't open, check the terminal for a URL)")
    print()

    # Run local server to capture OAuth redirect
    credentials = flow.run_local_server(
        port=0,  # Random available port
        access_type="offline",
        prompt="consent",
    )

    print()
    print("=" * 70)
    print("SUCCESS! Add this to your .env file:")
    print("=" * 70)
    print()
    print(f"YT_REFRESH_TOKEN={credentials.refresh_token}")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
