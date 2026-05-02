#!/usr/bin/env python3
"""Test if YouTube API write operations are available (not rate limited)."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.yt_pl_ctr.youtube import YouTubeAPIError, YouTubeClient

try:
    yt = YouTubeClient.from_env()

    # Test write operation - create a private test playlist
    playlist_id = yt.create_playlist("_quota_test", "temporary test", "private")
    print(f"OK - Write operations available (created {playlist_id})")

    # Clean up - delete the test playlist
    yt._service.playlists().delete(id=playlist_id).execute()
    print("OK - Cleaned up test playlist")

    sys.exit(0)

except YouTubeAPIError as e:
    if "429" in str(e) or "rate" in str(e).lower():
        print("RATE LIMITED - Write operations blocked")
        sys.exit(1)
    else:
        print(f"ERROR: {e}")
        sys.exit(2)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(2)
