"""CLI entry points."""

from __future__ import annotations

from pathlib import Path

import click

from wikiquotes_tagger.config import CONFIG_DEFAULT_PATH, load_config
from wikiquotes_tagger import db


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False),
    default=str(CONFIG_DEFAULT_PATH),
    help="Path to config.toml file.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """wikiquotes-tagger: Parse and AI-tag Wikiquote quotes."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(Path(config_path))


@cli.command()
@click.pass_context
def download(ctx: click.Context) -> None:
    """Download the latest Wikiquote XML dump."""
    from wikiquotes_tagger.downloader import decompress_dump, download_dump

    config = ctx.obj["config"]
    config.data_dir.mkdir(parents=True, exist_ok=True)
    bz2_path = download_dump(config.data_dir)
    click.echo(f"Downloaded: {bz2_path}")
    xml_path = decompress_dump(bz2_path)
    size_gb = xml_path.stat().st_size / 1e9
    click.echo(f"Decompressed: {xml_path} ({size_gb:.1f} GB)")


@cli.command()
@click.pass_context
def parse(ctx: click.Context) -> None:
    """Parse XML dump into SQLite quotes database."""
    from wikiquotes_tagger.downloader import find_dump_file
    from wikiquotes_tagger.parser import parse_and_store

    config = ctx.obj["config"]
    xml_path = find_dump_file(config.data_dir)
    if xml_path is None:
        click.echo("No XML dump found in data/. Run 'wikiquotes-tagger download' first.")
        raise SystemExit(1)

    db.init_db(config.db_path)
    count = parse_and_store(xml_path, config.db_path)
    click.echo(f"Done. Inserted {count:,} quotes.")


@cli.command()
@click.option("--batch-size", type=int, default=None, help="Override config batch_size.")
@click.option("--limit", type=int, default=None, help="Max quotes to tag this run.")
@click.pass_context
def tag(ctx: click.Context, batch_size: int | None, limit: int | None) -> None:
    """Tag untagged quotes via AI API (resumable)."""
    from wikiquotes_tagger.tagger import tag_quotes

    config = ctx.obj["config"]
    db.init_db(config.db_path)
    count = tag_quotes(config, batch_size_override=batch_size, limit=limit)
    click.echo(f"Done. Tagged {count:,} quotes this session.")


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics and tagging progress."""
    config = ctx.obj["config"]
    db.init_db(config.db_path)
    conn = db.get_connection(config.db_path)
    s = db.get_stats(conn)
    conn.close()

    click.echo(f"Database: {config.db_path}")
    click.echo(f"Total quotes:  {s['total']:>8,}")
    click.echo(f"  Parsed:      {s['parsed']:>8,}")
    click.echo(f"  Tagged:      {s['tagged']:>8,}")
    if s["errored"] > 0:
        click.echo(f"  Errored:     {s['errored']:>8,}")
    if s["total"] > 0:
        pct = s["tagged"] / s["total"] * 100
        click.echo(f"  Progress:    {pct:>7.1f}%")
    if s["top_categories"]:
        click.echo("\nTop categories:")
        for cat, count in s["top_categories"]:
            click.echo(f"  {cat:<30s} {count:>6,}")


@cli.command()
@click.pass_context
def export(ctx: click.Context) -> None:
    """Export tagged quotes (not yet implemented)."""
    click.echo("Export is not yet implemented. Use the SQLite database directly.")
    click.echo(f"Database path: {ctx.obj['config'].db_path}")
