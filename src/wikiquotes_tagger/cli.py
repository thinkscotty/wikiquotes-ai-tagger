"""CLI entry points."""

from __future__ import annotations

from pathlib import Path

import click

from wikiquotes_tagger.config import CONFIG_DEFAULT_PATH, SCORING_PROMPT_DEFAULT_PATH, load_config, load_scoring_config
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
@click.option("--sample", type=int, default=None, help="Tag N randomly-selected quotes (for testing prompt quality with diverse authors).")
@click.option("--debug", is_flag=True, default=False, help="Print raw API responses for debugging.")
@click.pass_context
def tag(ctx: click.Context, batch_size: int | None, limit: int | None, sample: int | None, debug: bool) -> None:
    """Tag untagged quotes via AI API (resumable).

    Use --sample N to tag N quotes chosen at random from throughout the database,
    avoiding the author skew of sequential processing.
    """
    from wikiquotes_tagger.tagger import tag_quotes

    config = ctx.obj["config"]
    db.init_db(config.db_path)
    count = tag_quotes(config, batch_size_override=batch_size, limit=limit, sample=sample, debug=debug)
    click.echo(f"Done. Tagged {count:,} quotes this session.")


@cli.command()
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def reset(ctx: click.Context, confirm: bool) -> None:
    """Reset all tagged/errored quotes to 'parsed' for re-tagging."""
    config = ctx.obj["config"]
    db.init_db(config.db_path)
    conn = db.get_connection(config.db_path)

    tagged = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'tagged'").fetchone()[0]
    errored = conn.execute("SELECT COUNT(*) FROM quotes WHERE status = 'error'").fetchone()[0]
    total = tagged + errored

    if total == 0:
        click.echo("No tagged or errored quotes to reset.")
        conn.close()
        return

    if not confirm:
        msg = f"Reset {tagged:,} tagged"
        if errored > 0:
            msg += f" and {errored:,} errored"
        msg += " quotes to 'parsed'?"
        click.confirm(msg, abort=True)

    tagged_count, errored_count = db.reset_tagged(conn)
    conn.close()

    click.echo(f"Reset {tagged_count:,} tagged quotes to 'parsed'.")
    if errored_count > 0:
        click.echo(f"Reset {errored_count:,} errored quotes to 'parsed'.")
    click.echo("Run 'wikiquotes-tagger tag' to re-tag all quotes.")


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics and tagging progress."""
    config = ctx.obj["config"]
    db.init_db(config.db_path)
    conn = db.get_connection(config.db_path)
    s = db.get_stats(conn)
    scoring = db.get_scoring_stats(conn)
    conn.close()

    click.echo(f"Database: {config.db_path}")
    click.echo(f"Total quotes:  {s['total']:>8,}")
    click.echo(f"  Parsed:      {s['parsed']:>8,}")
    click.echo(f"  Tagged:      {s['tagged']:>8,}")
    if s["errored"] > 0:
        click.echo(f"  Errored:     {s['errored']:>8,}")
    if s.get("non_english", 0) > 0:
        click.echo(f"  Non-English: {s['non_english']:>8,}")
    if s["total"] > 0:
        pct = s["tagged"] / s["total"] * 100
        click.echo(f"  Progress:    {pct:>7.1f}%")
    if s["top_categories"]:
        click.echo("\nTop categories:")
        for cat, count in s["top_categories"]:
            click.echo(f"  {cat:<30s} {count:>6,}")

    # Show scoring stats if any scores exist
    if scoring["scored"] > 0:
        click.echo(f"\nScoring:")
        click.echo(f"  Scored:      {scoring['scored']:>8,} of {scoring['tagged']:,} tagged")
        click.echo(f"  Avg quality: {scoring['avg_quality']}")


@cli.command()
@click.pass_context
def export(ctx: click.Context) -> None:
    """Export tagged quotes (not yet implemented)."""
    click.echo("Export is not yet implemented. Use the SQLite database directly.")
    click.echo(f"Database path: {ctx.obj['config'].db_path}")


@cli.command()
@click.option("--batch-size", type=int, default=None, help="Override default batch size (default: 20).")
@click.option("--limit", type=int, default=None, help="Max quotes to score this run.")
@click.option("--sample", type=int, default=None, help="Score N randomly-selected quotes.")
@click.option("--debug", is_flag=True, default=False, help="Print raw API responses for debugging.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=False),
    default=str(SCORING_PROMPT_DEFAULT_PATH),
    help="Path to scoring prompt TOML file.",
)
@click.pass_context
def score(
    ctx: click.Context,
    batch_size: int | None,
    limit: int | None,
    sample: int | None,
    debug: bool,
    prompt_file: str,
) -> None:
    """Score tagged quotes for quality (1-10) via AI API (resumable).

    Uses a separate prompt config (scoring_prompt.toml) from tagging.
    Scores only quotes that are already tagged but have no quality score.
    """
    from wikiquotes_tagger.scorer import score_quotes

    config = ctx.obj["config"]
    db.init_db(config.db_path)
    scoring_config = load_scoring_config(Path(prompt_file))
    count = score_quotes(
        config,
        scoring_config,
        batch_size_override=batch_size,
        limit=limit,
        sample=sample,
        debug=debug,
    )
    click.echo(f"Done. Scored {count:,} quotes this session.")


@cli.command("score-stats")
@click.pass_context
def score_stats(ctx: click.Context) -> None:
    """Show quality scoring progress and distribution."""
    config = ctx.obj["config"]
    db.init_db(config.db_path)
    conn = db.get_connection(config.db_path)
    s = db.get_scoring_stats(conn)
    conn.close()

    click.echo(f"Database: {config.db_path}")
    click.echo(f"Tagged quotes:  {s['tagged']:>8,}")
    click.echo(f"  Scored:       {s['scored']:>8,}")
    click.echo(f"  Unscored:     {s['unscored']:>8,}")
    if s["tagged"] > 0:
        pct = s["scored"] / s["tagged"] * 100
        click.echo(f"  Progress:     {pct:>7.1f}%")
    if s["avg_quality"] is not None:
        click.echo(f"\nAvg quality: {s['avg_quality']}")
    if s["distribution"]:
        click.echo("\nScore distribution:")
        total = s["scored"]
        for score_val in range(1, 11):
            count = s["distribution"].get(score_val, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            click.echo(f"  {score_val:>2}: {count:>7,} ({pct:>5.1f}%) {bar}")


@cli.command("reset-scores")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def reset_scores(ctx: click.Context, confirm: bool) -> None:
    """Reset all quality scores to NULL for re-scoring."""
    config = ctx.obj["config"]
    db.init_db(config.db_path)
    conn = db.get_connection(config.db_path)

    scored = conn.execute(
        "SELECT COUNT(*) FROM quotes WHERE quality IS NOT NULL"
    ).fetchone()[0]

    if scored == 0:
        click.echo("No scored quotes to reset.")
        conn.close()
        return

    if not confirm:
        click.confirm(f"Reset quality scores on {scored:,} quotes?", abort=True)

    count = db.reset_scores(conn)
    conn.close()
    click.echo(f"Reset quality scores on {count:,} quotes.")
