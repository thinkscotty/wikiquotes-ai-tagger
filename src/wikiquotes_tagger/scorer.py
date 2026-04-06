"""Batch AI quality scoring of tagged quotes."""

from __future__ import annotations

import json
import logging
import re
import signal
import time
from dataclasses import dataclass

import click
import httpx

from wikiquotes_tagger import db
from wikiquotes_tagger.config import AppConfig, ScoringPromptConfig
from wikiquotes_tagger.tagger import _call_ai_api

log = logging.getLogger(__name__)

# Flag for graceful Ctrl+C handling (separate from tagger's flag)
_interrupted = False


def _handle_sigint(signum: int, frame: object) -> None:
    """Set interrupt flag on first Ctrl+C. Exit on second."""
    global _interrupted
    if _interrupted:
        raise SystemExit(1)
    _interrupted = True
    click.echo("\nInterrupt received. Finishing current batch then exiting...")


@dataclass
class ScoreResult:
    quote_id: int
    score: int


def score_quotes(
    config: AppConfig,
    scoring_config: ScoringPromptConfig,
    *,
    batch_size_override: int | None = None,
    limit: int | None = None,
    sample: int | None = None,
    debug: bool = False,
) -> int:
    """Main entry point: score tagged quotes in batches. Returns count scored."""
    global _interrupted
    _interrupted = False

    batch_size = batch_size_override or 20
    conn = db.get_connection(config.db_path)
    total_scored = 0

    # Install graceful interrupt handler
    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    client = httpx.Client(
        base_url=config.api.base_url,
        headers={"Authorization": f"Bearer {config.api.api_key}"},
        timeout=httpx.Timeout(config.api.timeout_seconds, connect=10.0),
    )

    # If sample mode, pre-select random IDs from the unscored pool
    sample_ids: list[int] | None = None
    if sample is not None and sample > 0:
        sample_ids = db.get_random_unscored_ids(conn, sample)
        if not sample_ids:
            click.echo("No unscored tagged quotes found.")
            return 0
        click.echo(f"Sample mode: selected {len(sample_ids):,} random quotes from unscored pool")

    try:
        # Count total unscored for progress display
        unscored_total = conn.execute(
            "SELECT COUNT(*) FROM quotes WHERE status = 'tagged' AND quality IS NULL"
        ).fetchone()[0]

        if unscored_total == 0:
            click.echo("No unscored tagged quotes found.")
            return 0

        if sample_ids is not None:
            effective_limit = len(sample_ids)
        else:
            effective_limit = min(limit, unscored_total) if limit else unscored_total
        total_batches = (effective_limit + batch_size - 1) // batch_size
        click.echo(
            f"Scoring {effective_limit:,} of {unscored_total:,} unscored quotes "
            f"in ~{total_batches:,} batches of {batch_size}"
        )

        sample_offset = 0
        batch_num = 0
        consecutive_failures = 0

        while True:
            if _interrupted:
                click.echo("Interrupted. Progress saved.")
                break

            if sample_ids is not None:
                if sample_offset >= len(sample_ids):
                    break
                chunk_end = min(sample_offset + batch_size, len(sample_ids))
                chunk_ids = sample_ids[sample_offset:chunk_end]
                quotes = db.get_quotes_by_ids(conn, chunk_ids)
                sample_offset = chunk_end
            else:
                if limit is not None and total_scored >= limit:
                    break
                current_batch_size = batch_size
                if limit is not None:
                    current_batch_size = min(batch_size, limit - total_scored)
                quotes = db.get_unscored_batch(conn, current_batch_size)

            if not quotes:
                break

            batch_num += 1
            start_time = time.monotonic()

            # Build prompt
            system_msg, user_msg = _build_scoring_prompt(scoring_config, quotes)

            # Call AI API (reuse tagger's retry logic)
            try:
                response_text = _call_ai_api(client, config, system_msg, user_msg)
            except Exception as e:
                consecutive_failures += 1
                click.echo(f"  Batch {batch_num}: API error: {e} — skipping batch")
                log.warning("API error on scoring batch %d: %s", batch_num, e)
                if consecutive_failures >= config.api.max_retries:
                    click.echo(
                        f"  Skipping {len(quotes)} quotes after "
                        f"{consecutive_failures} consecutive failures"
                    )
                    consecutive_failures = 0
                time.sleep(config.api.delay_between_batches)
                continue

            consecutive_failures = 0

            # Parse response
            quote_ids = [q["id"] for q in quotes]

            if debug:
                click.echo(f"\n--- DEBUG: Raw scoring response (batch {batch_num}) ---")
                click.echo(response_text[:2000])
                click.echo(f"--- END DEBUG (quote_ids: {quote_ids[:5]}...) ---\n")

            results = _parse_score_response(response_text, quote_ids)

            # Apply scores to database
            scored_count = _apply_scores(conn, results)
            conn.commit()
            total_scored += scored_count

            elapsed = time.monotonic() - start_time
            skipped = len(quotes) - scored_count
            skip_msg = f", {skipped} skipped" if skipped > 0 else ""
            click.echo(
                f"  Batch {batch_num}/{total_batches}: "
                f"scored {scored_count} quotes{skip_msg} [{elapsed:.1f}s]"
            )

            if scored_count == 0:
                consecutive_failures += 1
                if consecutive_failures >= config.api.max_retries:
                    click.echo(
                        f"  Skipping {len(quotes)} quotes after "
                        f"{consecutive_failures} consecutive zero-result batches"
                    )
                    consecutive_failures = 0

            # Delay between batches
            if not _interrupted and config.api.delay_between_batches > 0:
                time.sleep(config.api.delay_between_batches)

    finally:
        client.close()
        conn.close()
        signal.signal(signal.SIGINT, old_handler)

    return total_scored


def _build_scoring_prompt(
    scoring_config: ScoringPromptConfig,
    quotes: list[dict],
) -> tuple[str, str]:
    """Build system and user messages for quality scoring.

    Injects calibration examples into the system prompt and formats
    the quote batch into the user prompt.
    """
    # Format calibration examples
    cal_lines = []
    for ex in scoring_config.calibration:
        cal_lines.append(
            f'- "{ex.text}" — {ex.author} → Score: {ex.score} ({ex.reason})'
        )
    calibration_text = "\n".join(cal_lines) if cal_lines else "No calibration examples provided."

    system_msg = scoring_config.system_prompt.replace("{{calibration_examples}}", calibration_text)

    # Format quotes for user prompt
    lines = []
    for i, q in enumerate(quotes, 1):
        author = q.get("author", "Unknown")
        text = q.get("text", "")
        lines.append(f'{i}. "{text}" — {author}')

    quotes_text = "\n".join(lines)
    user_msg = scoring_config.user_prompt_template.replace("{{quotes}}", quotes_text)

    return system_msg, user_msg


def _parse_score_response(response_text: str, quote_ids: list[int]) -> list[ScoreResult]:
    """Parse the AI JSON response into ScoreResult objects.

    Expected format: [{"id": 1, "score": 7}, {"id": 2, "score": 4}, ...]
    Maps 1-based 'id' from AI response back to actual quote_ids.
    """
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Find JSON array bounds
    start = text.find("[")
    end = text.rfind("]")

    if start == -1:
        log.warning("No JSON array found in scoring response")
        return []

    if end == -1 or end <= start:
        json_text = text[start:]
    else:
        json_text = text[start : end + 1]

    try:
        items = json.loads(json_text)
    except json.JSONDecodeError:
        # Salvage truncated response
        salvage = text[start:]
        last_brace = salvage.rfind("}")
        if last_brace == -1:
            log.warning("No complete JSON objects in scoring response")
            return []
        json_text = salvage[: last_brace + 1].rstrip(", \n\t") + "]"
        json_text = re.sub(r",\s*]", "]", json_text)
        try:
            items = json.loads(json_text)
        except json.JSONDecodeError as e:
            log.warning("Failed to parse scoring JSON response: %s", e)
            return []

    if not isinstance(items, list):
        log.warning("Expected JSON array, got %s", type(items).__name__)
        return []

    results: list[ScoreResult] = []
    seen_ids: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue

        # Extract and validate id (1-based index into the batch)
        raw_id = item.get("id")
        if raw_id is None:
            continue
        try:
            idx = int(raw_id) - 1  # Convert to 0-based
        except (ValueError, TypeError):
            continue
        if idx < 0 or idx >= len(quote_ids):
            continue

        if idx in seen_ids:
            continue
        seen_ids.add(idx)

        # Extract and validate score
        raw_score = item.get("score")
        if raw_score is None:
            continue
        try:
            score = int(round(float(raw_score)))
        except (ValueError, TypeError):
            continue

        # Clamp to valid range
        score = max(1, min(10, score))

        results.append(ScoreResult(
            quote_id=quote_ids[idx],
            score=score,
        ))

    return results


def _apply_scores(
    conn: db.sqlite3.Connection,
    results: list[ScoreResult],
) -> int:
    """Update quotes in the database with quality scores. Returns count updated."""
    count = 0
    for result in results:
        if db.update_quality(conn, result.quote_id, result.score):
            count += 1
    return count
