"""Typer CLI for yt-pl-ctr."""

import logging
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
import typer

# Load .env file if present
load_dotenv()
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .config import load_config
from .queue import VideoQueue
from .fetcher_queue import fetch_to_queue
from .processor import process_pending, watch_and_process
from .sync import classify_channel_videos, sync_all_channels
from .youtube import YouTubeClient, YouTubeAPIError

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


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
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
        Optional[int],
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
        Optional[list[str]],
        typer.Option("--channel", help="Only sync specific channel URL(s)"),
    ] = None,
) -> None:
    """Sync videos from configured channels to categorized playlists."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    config = load_config(config_path)
    console.print(f"[blue]Loaded config with {len(config.channels)} channel(s)[/blue]")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")

    try:
        youtube = YouTubeClient.from_env()
        channel_info = youtube.get_channel_info()
        console.print(f"[green]Authenticated as: {channel_info.get('title')} ({channel_info.get('custom_url')})[/green]")
    except YouTubeAPIError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Set YT_CLIENT_ID, YT_CLIENT_SECRET, and YT_REFRESH_TOKEN environment variables[/dim]")
        raise typer.Exit(1)

    stats = sync_all_channels(
        config=config,
        youtube=youtube,
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
    table.add_column("Skipped", justify="right", style="yellow")
    table.add_column("Errors", justify="right", style="red")

    for ch_stats in stats.channels:
        table.add_row(
            ch_stats.channel_url[:40],
            str(ch_stats.videos_processed),
            str(ch_stats.videos_added),
            str(ch_stats.videos_skipped),
            str(ch_stats.errors),
        )

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{stats.total_processed}[/bold]",
        f"[bold]{stats.total_added}[/bold]",
        f"[bold]{stats.total_skipped}[/bold]",
        f"[bold]{stats.total_errors}[/bold]",
    )
    console.print(table)

    if stats.total_errors > 0:
        raise typer.Exit(1)


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
        Optional[str],
        typer.Option("--channel", help="Only classify specific channel URL"),
    ] = None,
) -> None:
    """Preview video classifications without syncing to playlists."""
    setup_logging(False)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    config = load_config(config_path)

    for channel_config in config.channels:
        if channel_url and channel_config.url != channel_url:
            continue

        console.print(f"\n[blue]Channel: {channel_config.playlist_prefix or channel_config.url}[/blue]")
        console.print(f"[dim]URL: {channel_config.url}[/dim]")

        table = Table()
        table.add_column("Video Title", max_width=50)
        table.add_column("Guest", max_width=20)
        table.add_column("Category", style="cyan")
        table.add_column("Match", style="dim")

        results = classify_channel_videos(channel_config, limit=limit)

        for r in results:
            guest = r.video.title.split(" - ")[-1][:20] if " - " in r.video.title else "-"
            table.add_row(
                r.video.title[:50],
                guest,
                r.classification.category_name,
                f"{r.classification.match_reason}: {r.classification.matched_value or 'n/a'}"[:30],
            )

        console.print(table)

        # Category summary
        categories: dict[str, int] = {}
        for r in results:
            key = r.classification.category_key
            categories[key] = categories.get(key, 0) + 1

        console.print("\n[dim]Category distribution:[/dim]")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            console.print(f"  {cat}: {count}")


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
        raise typer.Exit(1)

    config = load_config(config_path)

    for channel in config.channels:
        console.print(f"\n[bold cyan]{channel.playlist_prefix or 'Channel'}[/bold cyan]")
        console.print(f"  URL: {channel.url}")
        console.print(f"  Default category: {channel.default_category}")
        console.print("  Categories:")
        for key, cat in channel.categories.items():
            console.print(f"    [green]{key}[/green]: {cat.name}")
            if cat.guests:
                console.print(f"      Guests: {len(cat.guests)}")
            if cat.keywords:
                console.print(f"      Keywords: {len(cat.keywords)}")


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
        raise typer.Exit(1)


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
        Optional[int],
        typer.Option("--limit", "-l", help="Max videos per channel"),
    ] = 50,
    offset: Annotated[
        Optional[int],
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
) -> None:
    """Fetch video metadata and add to queue (for backfilling)."""
    from .fetcher_queue import load_state, save_state, get_channel_offset, STATE_FILE

    setup_logging(verbose)

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

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

    count = fetch_to_queue(config, queue, limit=limit, offset=offset, delay=delay, resume=resume)

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
        Optional[int],
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
        raise typer.Exit(1)

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
            raise typer.Exit(1)

    if watch:
        console.print("[cyan]Watch mode - press Ctrl+C to stop[/cyan]")
        watch_and_process(config, queue, youtube, dry_run=dry_run)
    else:
        processed, succeeded, failed = process_pending(
            config, queue, youtube, dry_run=dry_run, limit=limit
        )
        console.print(f"\n[green]Processed: {processed}, Succeeded: {succeeded}, Failed: {failed}[/green]")


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
