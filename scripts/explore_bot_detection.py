#!/usr/bin/env python3
"""Explore yt-dlp bot detection on GitHub Actions-like environments.

Tests different yt-dlp options to understand what triggers bot detection
and what (if anything) bypasses it. Run locally first, then in CI.

Usage:
    uv run python scripts/explore_bot_detection.py
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

# A small public video that's always available
TEST_VIDEO_ID = "dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up (stable, public)
TEST_CHANNEL_URL = "https://www.youtube.com/@joerogan/videos"


@dataclass
class ProbeResult:
    name: str
    success: bool
    error: str = ""
    title: str = ""
    duration_ms: float = 0.0


def run_ydlp(*args, timeout=30) -> tuple[bool, str, str]:
    """Run yt-dlp with given args, return (success, stdout, stderr)."""
    cmd = ["yt-dlp", *args]
    print(f"  cmd: {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    print(f"  exit={result.returncode} in {elapsed:.1f}s")
    return result.returncode == 0, result.stdout, result.stderr


def probe_video(name: str, extra_args: list[str]) -> ProbeResult:
    """Try fetching metadata for a single known video with given yt-dlp args."""
    print(f"\n[{name}]")
    t0 = time.time()
    ok, stdout, stderr = run_ydlp(
        "--skip-download",
        "--dump-json",
        "--no-warnings",
        *extra_args,
        f"https://www.youtube.com/watch?v={TEST_VIDEO_ID}",
    )
    elapsed = (time.time() - t0) * 1000

    if not ok:
        err = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
        print(f"  FAIL: {err[:120]}")
        return ProbeResult(name=name, success=False, error=err, duration_ms=elapsed)

    try:
        data = json.loads(stdout)
        title = data.get("title", "?")
        print(f"  OK: {title}")
        return ProbeResult(name=name, success=True, title=title, duration_ms=elapsed)
    except json.JSONDecodeError:
        return ProbeResult(name=name, success=False, error="bad JSON", duration_ms=elapsed)


def probe_flat_channel(name: str, extra_args: list[str], limit: int = 5) -> ProbeResult:
    """Try flat-playlist extraction (just video IDs) from the channel."""
    print(f"\n[{name} - flat channel]")
    t0 = time.time()
    ok, stdout, stderr = run_ydlp(
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        f"--playlist-end={limit}",
        *extra_args,
        TEST_CHANNEL_URL,
    )
    elapsed = (time.time() - t0) * 1000

    if not ok or not stdout.strip():
        err = stderr.strip().splitlines()[-1] if stderr.strip() else "no output"
        print(f"  FAIL: {err[:120]}")
        return ProbeResult(name=name, success=False, error=err, duration_ms=elapsed)

    count = len([ln for ln in stdout.strip().splitlines() if ln])
    print(f"  OK: got {count} entries")
    return ProbeResult(name=name, success=True, title=f"{count} entries", duration_ms=elapsed)


def check_version():
    print("=" * 60)
    print("yt-dlp version check")
    print("=" * 60)
    ok, stdout, _ = run_ydlp("--version")
    if ok:
        print(f"  installed: {stdout.strip()}")
    print("  pinned in pyproject.toml: >=2025.12.8")
    print("  latest stable: 2026.3.17  (as of 2026-04-30)")
    print()


def check_environment():
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    relevant = ["CI", "GITHUB_ACTIONS", "RUNNER_OS", "YT_COOKIES_FROM", "YT_COOKIES_FILE"]
    for var in relevant:
        val = os.environ.get(var, "(not set)")
        print(f"  {var}={val}")
    print()


def main():
    check_version()
    check_environment()

    results: list[ProbeResult] = []

    print("=" * 60)
    print("Probe 1: Baseline — no extra args")
    print("=" * 60)
    results.append(probe_video("baseline", []))

    print("\n" + "=" * 60)
    print("Probe 2: With --extractor-retries 1 --sleep-requests 2")
    print("=" * 60)
    results.append(
        probe_video("sleep+retry", ["--extractor-retries", "1", "--sleep-requests", "2"])
    )

    print("\n" + "=" * 60)
    print("Probe 3: With --user-agent (Chrome UA)")
    print("=" * 60)
    chrome_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    results.append(probe_video("chrome-ua", ["--user-agent", chrome_ua]))

    print("\n" + "=" * 60)
    print("Probe 4: Flat channel (known to work from CI logs)")
    print("=" * 60)
    results.append(probe_flat_channel("flat-channel", []))

    print("\n" + "=" * 60)
    print("Probe 5: Cookies from file (if YT_COOKIES_FILE is set)")
    print("=" * 60)
    cookies_file = os.environ.get("YT_COOKIES_FILE")
    if cookies_file and os.path.exists(cookies_file):
        results.append(probe_video("cookies-file", ["--cookies", cookies_file]))
    else:
        print("  SKIP: YT_COOKIES_FILE not set or file not found")
        print(
            "  To export cookies: yt-dlp --cookies-from-browser firefox --cookies cookies.txt <url>"
        )

    print("\n" + "=" * 60)
    print("Probe 6: PO token (requires yt-dlp >= 2024.11)")
    print("=" * 60)
    po_token = os.environ.get("YT_PO_TOKEN")
    visitor_data = os.environ.get("YT_VISITOR_DATA")
    if po_token and visitor_data:
        results.append(
            probe_video(
                "po-token",
                [
                    "--extractor-args",
                    f"youtube:po_token=web+{po_token};visitor_data={visitor_data}",
                ],
            )
        )
    else:
        print("  SKIP: YT_PO_TOKEN / YT_VISITOR_DATA not set")
        print("  See: https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        status = "OK " if r.success else "FAIL"
        detail = r.title if r.success else r.error[:60]
        print(f"  [{status}] {r.name:<20} {detail}")

    any_ok = any(r.success for r in results if "flat" not in r.name)
    if not any_ok:
        print("\nConclusion: All per-video fetches blocked. Likely causes:")
        print("  1. Outdated yt-dlp (pinned 2025.12.8, latest 2026.3.17)")
        print("  2. GitHub Actions IP range flagged by YouTube")
        print("  3. No cookies/PO token provided")
        print("\nNext steps to try:")
        print("  - Update yt-dlp: change >=2025.12.8 to >=2026.3.17 in pyproject.toml")
        print(
            "  - Export cookies: yt-dlp --cookies-from-browser firefox --cookies cookies.txt <url>"
        )
        print("  - Add cookies.txt as GitHub secret and pass via YT_COOKIES_FILE")
        print("  - Consider YouTube Data API v3 (no bot detection, uses OAuth)")

    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
