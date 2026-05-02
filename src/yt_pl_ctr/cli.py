"""Typer CLI for yt-pl-ctr."""

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .config import load_config
from .fetcher import VideoFetcherProtocol, YouTubeAPIFetcher, YtDlpFetcher, is_ci
from .fetcher_queue import fetch_to_queue
from .processor import process_pending, watch_and_process
from .queue import VideoQueue
from .sync import classify_channel_videos, sync_all_channels
from .youtube import YouTubeAPIError, YouTubeClient

# Load .env before any code that reads env vars
load_dotenv()

app = typer.Typer(
    name="yt-pl-ctr",
    help="YouTube Playlist Controller - Classify videos and manage playlists automatically",
    no_args_is_help=True,
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"yt-pl-ctr {__version__}")
        raise typer.Exit()


def _build_fetcher(youtube: YouTubeClient | None, mode: str = "auto") -> VideoFetcherProtocol:
    """Select fetcher implementation.

    mode: 'auto', 'api' (force), 'ytdlp' (force).
    CLI --fetcher flag > YT_FETCHER env var > auto logic.

    Auto logic:
      - GitHub Actions (GITHUB_ACTIONS=true): API fetcher — yt-dlp is bot-blocked on CI IPs
      - Residential / local: yt-dlp — saves API quota, works fine on non-CI IPs
    Playlist CRUD always uses the YouTube Data API regardless of fetcher choice.
    """
    resolved = mode if mode != "auto" else os.environ.get("YT_FETCHER", "auto")
    if resolved == "ytdlp":
        return YtDlpFetcher()
    if resolved == "api":
        if youtube is None:
            raise YouTubeAPIError("YT_FETCHER=api but no YouTube credentials found")
        return YouTubeAPIFetcher(youtube)
    # auto: use environment signal to pick the right backend
    if is_ci():
        if youtube is None:
            raise YouTubeAPIError("Running on GitHub Actions but no YouTube credentials found")
        return YouTubeAPIFetcher(youtube)
    return YtDlpFetcher()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", "-V", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """YouTube Playlist Controller."""
    pass


@app.command()
def sync(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file (YAML)"),
    ],
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Max videos per channel (overrides config)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show what would be done without making changes"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    channel: Annotated[
        list[str] | None,
        typer.Option("--channel", help="Only sync specific channel URL(s)"),
    ] = None,
    fetcher: Annotated[
        str,
        typer.Option(
            "--fetcher", help="Fetcher backend: auto, api, ytdlp (overrides YT_FETCHER env var)"
        ),
    ] = "auto",
) -> None:
    """Sync videos from configured channels to categorized playlists."""
    setup_logging(verbose)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)  # noqa: B904

    config = load_config(config_path)
    console.print(f"[blue]Loaded config with {len(config.channels)} channel(s)[/blue]")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")

    try:
        youtube = YouTubeClient.from_env()
        channel_info = youtube.get_channel_info()
        console.print(
            f"[green]Authenticated as: {channel_info.get('title')} "
            f"({channel_info.get('custom_url')})[/green]"
        )
    except YouTubeAPIError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Set YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN[/dim]")
        raise typer.Exit(1)  # noqa: B904

    stats = sync_all_channels(
        config=config,
        youtube=youtube,
        fetcher=_build_fetcher(youtube, mode=fetcher),
        limit=limit,
        dry_run=dry_run,
        channels=channel,
    )

    # Print summary
    console.print()
    table = Table(title="Sync Summary")
    table.add_column("Channel", style="cyan")
    table.add_column("Processed", justify="right")
    table.add_column("Added", justify="right", style="green")
    table.add_column("Reclassified", justify="right", style="blue")
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Errors", justify="right", style="red")

    for ch_stats in stats.channels:
        table.add_row(
            ch_stats.channel_url[:40],
            str(ch_stats.videos_processed),
            str(ch_stats.videos_added),
            str(ch_stats.videos_reclassified),
            str(ch_stats.videos_skipped),
            str(ch_stats.errors),
        )

    total_reclassified = sum(c.videos_reclassified for c in stats.channels)
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{stats.total_processed}[/bold]",
        f"[bold]{stats.total_added}[/bold]",
        f"[bold]{total_reclassified}[/bold]",
        f"[bold]{stats.total_skipped}[/bold]",
        f"[bold]{stats.total_errors}[/bold]",
    )
    console.print(table)

    if stats.total_errors > 0:
        raise typer.Exit(1)  # noqa: B904


def _print_classify_table(console: Console, clf, videos, channel_config) -> None:
    table = Table()
    table.add_column("Video Title", max_width=50)
    table.add_column("Guest", max_width=20)
    table.add_column("Category", style="cyan")
    table.add_column("Skip", justify="center")
    table.add_column("Match", style="dim")

    categories: dict[str, int] = {}
    for video in videos:
        result = clf.classify(video)
        guest = video.title.split(" - ")[-1][:20] if " - " in video.title else "-"
        skip_mark = "[red]✗[/red]" if result.skipped else "[green]✓[/green]"
        table.add_row(
            video.title[:50],
            guest,
            result.category_name,
            skip_mark,
            f"{result.match_reason}: {result.matched_value or 'n/a'}"[:35],
        )
        categories[result.category_key] = categories.get(result.category_key, 0) + 1

    console.print(table)
    console.print("\n[dim]Category distribution:[/dim]")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        console.print(f"  {cat}: {count}")


@app.command()
def classify(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file (YAML)"),
    ],
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Max videos per channel"),
    ] = 10,
    channel_url: Annotated[
        str | None,
        typer.Option("--channel", help="Only classify specific channel URL"),
    ] = None,
    video_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--video-id",
            "-v",
            help="Classify specific video ID(s) instead of fetching from channel",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Enable debug logging"),
    ] = False,
) -> None:
    """Preview video classifications without syncing to playlists."""
    setup_logging(verbose)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)  # noqa: B904

    config = load_config(config_path)

    # When targeting specific video IDs, fetch their metadata via YouTube API
    if video_ids:
        try:
            youtube = YouTubeClient.from_env()
        except YouTubeAPIError as e:
            console.print(f"[red]--video-id requires YouTube API credentials: {e}[/red]")
            raise typer.Exit(1)  # noqa: B904

        from .classifier import VideoClassifier

        videos = youtube.get_videos_metadata(video_ids)
        if not videos:
            console.print("[red]No videos found for given IDs[/red]")
            raise typer.Exit(1)  # noqa: B904

        for channel_config in config.channels:
            if channel_url and channel_config.url != channel_url:
                continue
            console.print(
                f"\n[blue]Channel: {channel_config.playlist_prefix or channel_config.url}[/blue]"
            )
            clf = VideoClassifier(channel_config)
            _print_classify_table(console, clf, videos, channel_config)
        return

    for channel_config in config.channels:
        if channel_url and channel_config.url != channel_url:
            continue

        console.print(
            f"\n[blue]Channel: {channel_config.playlist_prefix or channel_config.url}[/blue]"
        )
        console.print(f"[dim]URL: {channel_config.url}[/dim]")

        table = Table()
        table.add_column("Video Title", max_width=50)
        table.add_column("Guest", max_width=20)
        table.add_column("Category", style="cyan")
        table.add_column("Match", style="dim")

        from .classifier import VideoClassifier

        clf = VideoClassifier(channel_config)
        results = classify_channel_videos(channel_config, limit=limit, fetcher=_build_fetcher(None))
        videos = [r.video for r in results]
        _print_classify_table(console, clf, videos, channel_config)


@app.command()
def list_channels(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file (YAML)"),
    ],
) -> None:
    """List configured channels and their categories."""
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)  # noqa: B904

    config = load_config(config_path)

    for channel in config.channels:
        console.print(f"\n[bold cyan]{channel.playlist_prefix or 'Channel'}[/bold cyan]")
        console.print(f"  URL: {channel.url}")
        console.print(f"  Taxonomy leaf topics: {len(channel.all_leaf_slugs())}")
        console.print(f"  Playlists ({len(channel.playlists)}):")
        for slug, pl in channel.playlists.items():
            console.print(f"    [green]{slug}[/green]: {pl.title}")


@app.command()
def whoami() -> None:
    """Show authenticated YouTube account info."""
    try:
        youtube = YouTubeClient.from_env()
        info = youtube.get_channel_info()
        console.print(f"[green]Channel:[/green] {info.get('title')}")
        console.print(f"[green]Handle:[/green] {info.get('custom_url')}")
        console.print(f"[green]ID:[/green] {info.get('id')}")
    except YouTubeAPIError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)  # noqa: B904


# === Backfill commands (local, queue-based) ===


@app.command()
def fetch(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file (YAML)"),
    ],
    queue_dir: Annotated[
        Path,
        typer.Option("--queue", "-q", help="Queue directory"),
    ] = Path("queue"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Max videos per channel"),
    ] = 50,
    offset: Annotated[
        int | None,
        typer.Option("--offset", "-o", help="Skip first N videos (overrides saved offset)"),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", "-r", help="Resume from last saved offset"),
    ] = True,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Reset saved offsets to 0"),
    ] = False,
    delay: Annotated[
        float,
        typer.Option("--delay", "-d", help="Delay between fetches (seconds)"),
    ] = 1.0,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
    fetcher: Annotated[
        str,
        typer.Option(
            "--fetcher", help="Fetcher backend: auto, api, ytdlp (overrides YT_FETCHER env var)"
        ),
    ] = "auto",
) -> None:
    """Fetch video metadata and add to queue (for backfilling)."""
    from .fetcher_queue import STATE_FILE, get_channel_offset, load_state, save_state

    setup_logging(verbose)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)  # noqa: B904

    config = load_config(config_path)
    queue = VideoQueue(queue_dir)

    # Handle reset
    if reset:
        save_state({"offsets": {}}, Path(STATE_FILE))
        console.print("[yellow]Reset all offsets to 0[/yellow]")

    # Show current offsets
    state = load_state(Path(STATE_FILE))
    console.print(f"[blue]Fetching from {len(config.channels)} channel(s)[/blue]")
    for ch in config.channels:
        saved_offset = get_channel_offset(state, ch.url)
        if offset is not None:
            console.print(f"[dim]  {ch.playlist_prefix}: offset={offset} (override)[/dim]")
        elif resume and saved_offset > 0:
            console.print(f"[dim]  {ch.playlist_prefix}: offset={saved_offset} (resumed)[/dim]")
        else:
            console.print(f"[dim]  {ch.playlist_prefix}: offset=0[/dim]")
    console.print(f"[dim]Limit: {limit}, Queue: {queue_dir}[/dim]")

    try:
        youtube = YouTubeClient.from_env()
    except YouTubeAPIError:
        youtube = None

    count = fetch_to_queue(
        config,
        queue,
        fetcher=_build_fetcher(youtube, mode=fetcher),
        limit=limit,
        offset=offset,
        delay=delay,
        resume=resume,
    )

    console.print(f"\n[green]Fetched {count} videos[/green]")
    console.print(f"[dim]Pending: {queue.pending_count()}[/dim]")


@app.command()
def process(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to config file (YAML)"),
    ],
    queue_dir: Annotated[
        Path,
        typer.Option("--queue", "-q", help="Queue directory"),
    ] = Path("queue"),
    watch: Annotated[
        bool,
        typer.Option("--watch", "-w", help="Watch mode - run continuously"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Don't actually add to playlists"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Max items to process (non-watch mode)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Process queued videos - classify and add to playlists (for backfilling)."""
    setup_logging(verbose)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)  # noqa: B904

    config = load_config(config_path)
    queue = VideoQueue(queue_dir)

    console.print(f"[blue]Queue: {queue_dir}[/blue]")
    console.print(f"[dim]Pending: {queue.pending_count()}[/dim]")

    if dry_run:
        console.print("[yellow]DRY RUN MODE[/yellow]")

    youtube = None
    if not dry_run:
        try:
            youtube = YouTubeClient.from_env()
            info = youtube.get_channel_info()
            console.print(f"[green]YouTube: {info.get('title')}[/green]")
        except YouTubeAPIError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)  # noqa: B904

    if watch:
        console.print("[cyan]Watch mode - press Ctrl+C to stop[/cyan]")
        watch_and_process(config, queue, youtube, dry_run=dry_run)
    else:
        processed, succeeded, failed = process_pending(
            config, queue, youtube, dry_run=dry_run, limit=limit
        )
        console.print(
            f"\n[green]Processed: {processed}, Succeeded: {succeeded}, Failed: {failed}[/green]"
        )


@app.command()
def queue_status(
    queue_dir: Annotated[
        Path,
        typer.Option("--queue", "-q", help="Queue directory"),
    ] = Path("queue"),
) -> None:
    """Show queue status."""
    queue = VideoQueue(queue_dir)

    pending = len(list(queue.pending_dir.glob("*.json")))
    done = len(list(queue.done_dir.glob("*.json")))
    failed = len(list(queue.failed_dir.glob("*.json")))

    console.print(f"[cyan]Queue: {queue_dir}[/cyan]")
    console.print(f"  Pending: [yellow]{pending}[/yellow]")
    console.print(f"  Done:    [green]{done}[/green]")
    console.print(f"  Failed:  [red]{failed}[/red]")


def main_cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main_cli()
