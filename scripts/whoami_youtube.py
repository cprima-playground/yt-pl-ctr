#!/usr/bin/env python3
"""Check which YouTube account is authenticated."""

import os

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/youtube"]

creds = Credentials(
    token=None,
    refresh_token=os.environ["YT_REFRESH_TOKEN"],
    token_uri="https://oauth2.googleapis.com/token",
    client_id=os.environ["YT_CLIENT_ID"],
    client_secret=os.environ["YT_CLIENT_SECRET"],
    scopes=SCOPES,
)

yt = build("youtube", "v3", credentials=creds)
resp = yt.channels().list(part="snippet", mine=True).execute()

for ch in resp.get("items", []):
    s = ch["snippet"]
    print(s.get("title"), s.get("customUrl"))
