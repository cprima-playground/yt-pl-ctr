#!/usr/bin/env python3
"""Test whether upgrading yt-dlp resolves bot detection.

Installs the latest yt-dlp into a temp venv, then probes YouTube.
Run this locally or in CI to confirm whether the pinned version is the issue.

Usage:
    uv run python scripts/test_yt_dlp_upgrade.py
"""

import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

TEST_VIDEO_ID = "dQw4w9WgXcQ"  # stable public video
TEST_VIDEO_URL = f"https://www.youtube.com/watch?v={TEST_VIDEO_ID}"


def get_installed_version() -> str:
    result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def fetch_video_metadata(yt_dlp_cmd: str) -> tuple[bool, str]:
    """Fetch metadata for the test video. Returns (success, title_or_error)."""
    result = subprocess.run(
        [yt_dlp_cmd, "--skip-download", "--dump-json", "--no-warnings", TEST_VIDEO_URL],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "failed"
        return False, err
    try:
        data = json.loads(result.stdout)
        return True, data.get("title", "no title")
    except json.JSONDecodeError:
        return False, "bad JSON"


def test_with_version(label: str, yt_dlp_cmd: str):
    print(f"\n[{label}]")
    version_result = subprocess.run([yt_dlp_cmd, "--version"], capture_output=True, text=True)
    version = version_result.stdout.strip()
    print(f"  version: {version}")
    ok, detail = fetch_video_metadata(yt_dlp_cmd)
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {detail[:100]}")
    return ok


def main():
    print("=" * 60)
    print("yt-dlp version upgrade test")
    print("=" * 60)
    print(f"Test video: {TEST_VIDEO_URL}")

    # Test 1: Currently installed version
    current_version = get_installed_version()
    print(f"\nCurrently installed: {current_version}")
    ok_current = test_with_version("current install", "yt-dlp")

    # Test 2: Latest yt-dlp in a fresh temp venv
    print("\nInstalling latest yt-dlp in temp venv...")
    with tempfile.TemporaryDirectory() as tmpdir:
        venv_dir = Path(tmpdir) / "venv"
        venv.create(str(venv_dir), with_pip=True)

        pip = str(venv_dir / "bin" / "pip")
        yt_dlp_latest = str(venv_dir / "bin" / "yt-dlp")

        install = subprocess.run(
            [pip, "install", "--quiet", "yt-dlp"],
            capture_output=True,
            text=True,
        )
        if install.returncode != 0:
            print(f"  install failed: {install.stderr[:200]}")
            return 1

        ok_latest = test_with_version("latest yt-dlp", yt_dlp_latest)

    print("\n" + "=" * 60)
    print("Result")
    print("=" * 60)
    print(f"  current ({current_version}): {'OK' if ok_current else 'BLOCKED'}")
    print(f"  latest:                     {'OK' if ok_latest else 'BLOCKED'}")

    if not ok_current and ok_latest:
        print("\n  => Upgrading yt-dlp fixes bot detection.")
        print("     Change pyproject.toml: yt-dlp>=2025.12.8  →  yt-dlp>=2026.3.17")
        print("     Then run: uv lock --upgrade-package yt-dlp && uv sync")
    elif not ok_current and not ok_latest:
        print("\n  => Both versions blocked. IP-level block or cookies required.")
        print("     Consider: YouTube Data API v3, PO token, or cookies export.")
    elif ok_current:
        print("\n  => Current version already works (maybe environment-specific issue).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
