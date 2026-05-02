#!/usr/bin/env python3
"""Fetch raw Wikipedia data for each episode's guest and store in cache.

Stores episodes/{id}/wikipedia.json with raw {guest, found, summary, categories, url}.
No category mapping — raw text only, for use as ML features.

Usage:
    uv run python scripts/fetch_wikipedia.py
    uv run python scripts/fetch_wikipedia.py --limit 100
    uv run python scripts/fetch_wikipedia.py --video-id MtoPEub7XwA
    uv run python scripts/fetch_wikipedia.py --force   # re-fetch existing
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import wikipediaapi

import cache as cache_mod

GUEST_RE = re.compile(r"\s-\s(.+)$")
USER_AGENT = "yt-pl-ctr/0.1.0 (https://github.com/cprima-playground/yt-pl-ctr)"
_CATEGORY_NOISE = {"stub", "articles", "pages", "wikidata", "short description", "cs1", "use "}


def _default_cache_dir() -> Path:
    env = os.environ.get("YT_CACHE_DIR")
    return Path(env) if env else Path(__file__).parent.parent / "cache"


def _extract_guest(title: str) -> str | None:
    m = GUEST_RE.search(title or "")
    return m.group(1).strip() if m else None


def _is_disambiguation(page) -> bool:
    return any("disambiguation" in cat.lower() for cat in page.categories)


def _fetch_wikipedia(guest: str) -> dict:
    wiki = wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )
    page = wiki.page(guest)

    if not page.exists():
        return {
            "guest": guest,
            "found": False,
            "summary": "",
            "categories": [],
            "url": "",
            "alternatives": [],
        }

    # Disambiguation page — multiple people share this name
    if _is_disambiguation(page):
        # Extract linked page titles from the disambiguation summary as hints
        alternatives = re.findall(r"\[\[([^\]|]+)", page.text or "")[:8]
        return {
            "guest": guest,
            "found": False,
            "summary": "",
            "categories": [],
            "url": page.fullurl,
            "ambiguous": True,
            "alternatives": alternatives,
        }

    categories = [
        cat.replace("Category:", "")
        for cat in page.categories
        if not any(noise in cat.lower() for noise in _CATEGORY_NOISE)
    ][:30]

    # Flag if the page title doesn't closely match the guest name — possible wrong match
    title_lower = page.title.lower()
    guest_lower = guest.lower()
    name_match = any(part in title_lower for part in guest_lower.split())
    ambiguous = not name_match

    return {
        "guest": guest,
        "found": True,
        "summary": page.summary[:2000],
        "categories": categories,
        "url": page.fullurl,
        "ambiguous": ambiguous,
        "page_title": page.title,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch Wikipedia data for episode guests")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch even if wikipedia.json exists"
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or _default_cache_dir()
    if not cache_dir.exists():
        print(f"Cache not found: {cache_dir}")
        return 1

    index = cache_mod.read_index(cache_dir)
    if not index:
        print("Index is empty.")
        return 1

    if args.video_ids:
        targets = [e for e in index if e["video_id"] in args.video_ids]
    else:
        targets = [e for e in index if _extract_guest(e.get("title", ""))]
        if not args.force:
            targets = [e for e in targets if not cache_mod.has_wikipedia(cache_dir, e["video_id"])]

    if args.limit:
        targets = targets[: args.limit]

    print(f"Cache: {cache_dir}")
    print(f"Fetching Wikipedia for {len(targets)} episodes")
    print()

    fetched = 0
    not_found = 0
    no_guest = 0

    for i, entry in enumerate(targets, 1):
        vid = entry["video_id"]
        title = entry.get("title", "")
        guest = _extract_guest(title)

        if not guest:
            no_guest += 1
            continue

        print(f"  [{i}/{len(targets)}] {guest[:50]} ... ", end="", flush=True)

        try:
            data = _fetch_wikipedia(guest)
        except Exception as e:
            print(f"error: {e}")
            time.sleep(2)
            continue

        cache_mod.write_wikipedia(cache_dir, vid, data)

        if data["found"]:
            fetched += 1
            ambig = " [AMBIGUOUS — check manually]" if data.get("ambiguous") else ""
            print(f"ok ({len(data['summary'])} chars, {len(data['categories'])} categories){ambig}")
        else:
            not_found += 1
            alts = data.get("alternatives", [])
            if alts:
                print(f"not found — possible matches: {', '.join(alts[:5])}")
            else:
                print("not found")

        # Wikipedia API rate limit — be polite
        time.sleep(0.5)

    print(f"\nDone. Found: {fetched}, Not found: {not_found}, No guest: {no_guest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
