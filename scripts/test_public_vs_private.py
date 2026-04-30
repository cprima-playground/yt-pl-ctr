#!/usr/bin/env python3
"""Test if PUBLIC vs PRIVATE playlist creation has different rate limits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/youtube"]

# Build service directly (no wrapper)
creds = Credentials(
    token=None,
    refresh_token=os.environ["YT_REFRESH_TOKEN"],
    token_uri="https://oauth2.googleapis.com/token",
    client_id=os.environ["YT_CLIENT_ID"],
    client_secret=os.environ["YT_CLIENT_SECRET"],
    scopes=SCOPES,
)
service = build("youtube", "v3", credentials=creds)

def test_playlist(privacy: str) -> bool:
    """Test creating a playlist with given privacy. Returns True if successful."""
    print(f"\nTesting {privacy.upper()} playlist creation...")
    try:
        result = service.playlists().insert(
            part="snippet,status",
            body={
                "snippet": {"title": f"_test_{privacy}", "description": "test"},
                "status": {"privacyStatus": privacy},
            },
        ).execute()
        playlist_id = result["id"]
        print(f"  SUCCESS: Created {privacy} playlist {playlist_id}")

        # Clean up
        service.playlists().delete(id=playlist_id).execute()
        print(f"  Cleaned up")
        return True
    except HttpError as e:
        print(f"  FAILED: {e.resp.status} - {e.reason}")
        return False

# Test both
print("Direct API test (no retry wrapper)")
print("=" * 50)

private_ok = test_playlist("private")
public_ok = test_playlist("public")

print("\n" + "=" * 50)
print(f"PRIVATE: {'OK' if private_ok else 'FAILED'}")
print(f"PUBLIC:  {'OK' if public_ok else 'FAILED'}")

if private_ok and not public_ok:
    print("\n>>> PUBLIC playlists are rate limited differently!")
elif not private_ok and not public_ok:
    print("\n>>> Both are rate limited - API issue")
elif private_ok and public_ok:
    print("\n>>> Both work - issue is elsewhere in code")
