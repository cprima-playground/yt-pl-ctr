"""
One-shot script: list all playlists, then delete any whose title is in TITLES_TO_DELETE.
Run with --dry-run first to confirm, then without to execute.
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yt_pl_ctr.youtube import YouTubeClient

TITLES_TO_DELETE = [
    "CO: Politics / US Elections & Domestic Politics",
    "CO: Politics / Deep State & Censorship",
    "CO: Media / Celebrity & Culture",
    "CO: Society & Law / Lawsuits & Courts",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = YouTubeClient.from_env()

    playlists = client.list_my_playlists()
    print(f"Found {len(playlists)} playlists:")
    for p in sorted(playlists, key=lambda x: x["title"]):
        marker = " <-- DELETE" if p["title"] in TITLES_TO_DELETE else ""
        print(f"  [{p['item_count']:4d}] {p['title']}  ({p['id']}){marker}")

    to_delete = [p for p in playlists if p["title"] in TITLES_TO_DELETE]
    if not to_delete:
        print("\nNothing to delete.")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would delete {len(to_delete)} playlist(s).")
        return

    print()
    for p in to_delete:
        print(f"Deleting: {p['title']} ({p['id']}, {p['item_count']} videos) ...")
        client.delete_playlist(p["id"])
        print("  done.")


if __name__ == "__main__":
    main()
