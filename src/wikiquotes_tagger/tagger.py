"""Batch AI tagging of quotes with keywords and categories."""

from __future__ import annotations

import json
import logging
import re
import signal
import sqlite3
import time
from dataclasses import dataclass
from random import random

import click
import httpx

from wikiquotes_tagger import db
from wikiquotes_tagger.config import AppConfig

log = logging.getLogger(__name__)

# Flag for graceful Ctrl+C handling
_interrupted = False

# Valid author_type values
VALID_AUTHOR_TYPES = frozenset({"person", "work", "concept", "religious_text", "fictional"})


def _handle_sigint(signum: int, frame: object) -> None:
    """Set interrupt flag on first Ctrl+C. Exit on second."""
    global _interrupted
    if _interrupted:
        raise SystemExit(1)
    _interrupted = True
    click.echo("\nInterrupt received. Finishing current batch then exiting...")


@dataclass
class TagResult:
    quote_id: int
    keywords: list[str]
    categories: list[str]  # 1-3 categories from canonical list, best-fit first
    author_type: str  # "person", "work", "concept", "religious_text", "fictional"
    religious_sentiment: str | None  # JSON string or None


def tag_quotes(
    config: AppConfig,
    *,
    batch_size_override: int | None = None,
    limit: int | None = None,
    sample: int | None = None,
    debug: bool = False,
) -> int:
    """Main entry point: tag untagged quotes in batches. Returns count tagged.

    If `sample` is set, picks that many random quotes from the full untagged pool
    and tags only those (for testing prompt quality with diverse authors).
    """
    global _interrupted
    _interrupted = False

    batch_size = batch_size_override or config.api.batch_size
    conn = db.get_connection(config.db_path)
    batch_id = db.next_batch_id(conn)
    total_tagged = 0

    # Install graceful interrupt handler
    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    client = httpx.Client(
        base_url=config.api.base_url,
        headers={"Authorization": f"Bearer {config.api.api_key}"},
        timeout=httpx.Timeout(config.api.timeout_seconds, connect=10.0),
    )

    # If sample mode, pre-select random IDs from the full untagged pool
    sample_ids: list[int] | None = None
    if sample is not None and sample > 0:
        sample_ids = db.get_random_untagged_ids(conn, sample)
        if not sample_ids:
            click.echo("No untagged quotes found.")
            return 0
        click.echo(f"Sample mode: selected {len(sample_ids):,} random quotes from untagged pool")

    try:
        # Count total untagged for progress display
        untagged_total = conn.execute(
            "SELECT COUNT(*) FROM quotes WHERE status = 'parsed'"
        ).fetchone()[0]

        if untagged_total == 0:
            click.echo("No untagged quotes found.")
            return 0

        if sample_ids is not None:
            effective_limit = len(sample_ids)
        else:
            effective_limit = min(limit, untagged_total) if limit else untagged_total
        total_batches = (effective_limit + batch_size - 1) // batch_size
        click.echo(
            f"Tagging {effective_limit:,} of {untagged_total:,} untagged quotes "
            f"in ~{total_batches:,} batches of {batch_size}"
        )

        # For sample mode, chunk the pre-selected IDs into batches
        sample_offset = 0

        batch_num = 0
        consecutive_failures = 0
        while True:
            if _interrupted:
                click.echo("Interrupted. Progress saved.")
                break

            if sample_ids is not None:
                # Sample mode: pull next chunk from pre-selected IDs
                if sample_offset >= len(sample_ids):
                    break
                chunk_end = min(sample_offset + batch_size, len(sample_ids))
                chunk_ids = sample_ids[sample_offset:chunk_end]
                quotes = db.get_quotes_by_ids(conn, chunk_ids)
                sample_offset = chunk_end
            else:
                # Normal mode: sequential fetch
                if limit is not None and total_tagged >= limit:
                    break
                current_batch_size = batch_size
                if limit is not None:
                    current_batch_size = min(batch_size, limit - total_tagged)
                quotes = db.get_untagged_batch(conn, current_batch_size)

            if not quotes:
                break

            batch_num += 1
            start_time = time.monotonic()

            # Build prompt
            system_msg, user_msg = _build_prompt(config, quotes)

            # Call AI API
            try:
                response_text = _call_ai_api(client, config, system_msg, user_msg)
            except Exception as e:
                consecutive_failures += 1
                click.echo(f"  Batch {batch_num}: API error: {e} — skipping batch")
                log.warning("API error on batch %d: %s", batch_num, e)
                if consecutive_failures >= config.api.max_retries:
                    # Same batch keeps failing — mark quotes as 'error' to avoid infinite loop
                    for q in quotes:
                        conn.execute(
                            "UPDATE quotes SET status = 'error' WHERE id = ?",
                            (q["id"],),
                        )
                    conn.commit()
                    click.echo(
                        f"  Marked {len(quotes)} quotes as 'error' after "
                        f"{consecutive_failures} consecutive failures"
                    )
                    consecutive_failures = 0
                time.sleep(config.api.delay_between_batches)
                continue

            consecutive_failures = 0

            # Parse response
            quote_ids = [q["id"] for q in quotes]

            if debug:
                click.echo(f"\n--- DEBUG: Raw API response (batch {batch_num}) ---")
                click.echo(response_text[:2000])
                click.echo(f"--- END DEBUG (quote_ids: {quote_ids[:5]}...) ---\n")

            results = _parse_tag_response(response_text, quote_ids)

            # Apply tags to database
            tagged_count = _apply_tags(conn, results, batch_id)
            conn.commit()
            total_tagged += tagged_count

            elapsed = time.monotonic() - start_time
            skipped = len(quotes) - tagged_count
            skip_msg = f", {skipped} skipped" if skipped > 0 else ""
            click.echo(
                f"  Batch {batch_num}/{total_batches}: "
                f"tagged {tagged_count} quotes{skip_msg} [{elapsed:.1f}s]"
            )

            # Track consecutive zero-result batches (AI returned 200 but nothing usable)
            if tagged_count == 0:
                consecutive_failures += 1
                if consecutive_failures >= config.api.max_retries:
                    for q in quotes:
                        conn.execute(
                            "UPDATE quotes SET status = 'error' WHERE id = ?",
                            (q["id"],),
                        )
                    conn.commit()
                    click.echo(
                        f"  Marked {len(quotes)} quotes as 'error' after "
                        f"{consecutive_failures} consecutive zero-result batches"
                    )
                    consecutive_failures = 0

            batch_id += 1

            # Delay between batches (unless interrupted or done)
            if not _interrupted and config.api.delay_between_batches > 0:
                time.sleep(config.api.delay_between_batches)

    finally:
        client.close()
        conn.close()
        signal.signal(signal.SIGINT, old_handler)

    return total_tagged


def _build_prompt(config: AppConfig, quotes: list[dict]) -> tuple[str, str]:
    """Build system and user messages for the AI API call.

    Injects the category list into the system prompt via {{categories}} placeholder.
    Returns (system_message, user_message).
    """
    lines = []
    for i, q in enumerate(quotes, 1):
        author = q.get("author", "Unknown")
        text = q.get("text", "")
        lines.append(f'{i}. "{text}" \u2014 {author}')

    quotes_text = "\n".join(lines)
    user_msg = config.prompts.user_prompt_template.replace("{{quotes}}", quotes_text)

    # Inject categories into system prompt
    system_msg = config.prompts.system_prompt
    if config.categories:
        categories_str = ", ".join(config.categories)
        system_msg = system_msg.replace("{{categories}}", categories_str)

    return system_msg, user_msg


def _call_ai_api(
    client: httpx.Client,
    config: AppConfig,
    system_msg: str,
    user_msg: str,
) -> str:
    """Call the OpenAI-compatible chat completions endpoint.

    Retries on 429/5xx with exponential backoff.
    Returns the assistant message content string.
    """
    for attempt in range(config.api.max_retries + 1):
        try:
            body: dict = {
                "model": config.api.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            }
            if config.api.json_mode:
                body["response_format"] = {"type": "json_object"}

            response = client.post("/chat/completions", json=body)

            if response.status_code == 200:
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    raise RuntimeError(
                        f"Unexpected API response structure: {e} — "
                        f"keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
                    ) from e

            # Retriable errors
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < config.api.max_retries:
                    retry_after = response.headers.get("retry-after")
                    if retry_after and retry_after.isdigit():
                        wait = int(retry_after)
                    else:
                        wait = min(2**attempt + random(), 30.0)
                    log.warning(
                        "API returned %d, retrying in %.1fs (attempt %d/%d)",
                        response.status_code, wait, attempt + 1, config.api.max_retries,
                    )
                    time.sleep(wait)
                    continue

            # Non-retriable error
            response.raise_for_status()

        except httpx.TimeoutException:
            if attempt < config.api.max_retries:
                wait = min(2**attempt + random(), 30.0)
                log.warning("API timeout, retrying in %.1fs", wait)
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"API call failed after {config.api.max_retries} retries")


def _parse_tag_response(response_text: str, quote_ids: list[int]) -> list[TagResult]:
    """Parse the AI JSON response into TagResult objects.

    Handles markdown code fences, truncated responses, and malformed entries.
    Maps 1-based 'id' from AI response back to actual quote_ids.
    """
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Try parsing as a single JSON object first (some models return one object instead of an array)
    try:
        single = json.loads(text)
        if isinstance(single, dict) and "id" in single:
            # Wrap single object in a list
            text = json.dumps([single])
    except json.JSONDecodeError:
        pass

    # Find JSON array bounds
    start = text.find("[")
    end = text.rfind("]")

    if start == -1:
        log.warning("No JSON array found in response")
        return []

    if end == -1 or end <= start:
        json_text = text[start:]
    else:
        json_text = text[start : end + 1]

    try:
        items = json.loads(json_text)
    except json.JSONDecodeError:
        # Salvage truncated response: find last complete JSON object and close array
        salvage = text[start:]
        last_brace = salvage.rfind("}")
        if last_brace == -1:
            log.warning("No complete JSON objects in response")
            return []
        json_text = salvage[: last_brace + 1].rstrip(", \n\t") + "]"
        # Remove trailing comma before ] if present
        json_text = re.sub(r",\s*]", "]", json_text)
        try:
            items = json.loads(json_text)
        except json.JSONDecodeError as e:
            log.warning("Failed to parse JSON response: %s", e)
            return []

    if not isinstance(items, list):
        log.warning("Expected JSON array, got %s", type(items).__name__)
        return []

    results: list[TagResult] = []
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

        # Skip duplicate IDs — keep the first occurrence
        if idx in seen_ids:
            continue
        seen_ids.add(idx)

        # Extract keywords
        keywords = item.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        if not isinstance(keywords, list) or not keywords:
            continue
        keywords = [str(k).lower().strip() for k in keywords if k]
        keywords = [k for k in keywords if k]  # Remove empty strings after strip
        if not keywords:
            continue

        # Extract categories (supports both "categories" array and legacy "category" string)
        raw_categories = item.get("categories", item.get("category"))
        if isinstance(raw_categories, str):
            raw_categories = [raw_categories]  # wrap legacy single string
        if not isinstance(raw_categories, list) or not raw_categories:
            continue
        categories = [str(c).strip() for c in raw_categories if isinstance(c, str) and str(c).strip()]
        if not categories:
            continue
        categories = categories[:3]  # enforce max 3

        # Extract author_type (default to "person" if missing or invalid)
        author_type = item.get("author_type", "person")
        if not isinstance(author_type, str) or author_type not in VALID_AUTHOR_TYPES:
            author_type = "person"

        # Extract religious_sentiment (optional, only for religious quotes)
        religious_sentiment = None
        raw_sentiment = item.get("religious_sentiment")
        if raw_sentiment and isinstance(raw_sentiment, dict):
            # Validate: values must be "positive", "neutral", or "critical"
            valid_sentiments = {"positive", "neutral", "critical"}
            cleaned = {
                str(k).lower().strip(): str(v).lower().strip()
                for k, v in raw_sentiment.items()
                if str(v).lower().strip() in valid_sentiments
            }
            if cleaned:
                religious_sentiment = json.dumps(cleaned)

        results.append(TagResult(
            quote_id=quote_ids[idx],
            keywords=keywords,
            categories=categories,
            author_type=author_type,
            religious_sentiment=religious_sentiment,
        ))

    return results


def _apply_tags(
    conn: sqlite3.Connection,
    results: list[TagResult],
    batch_id: int,
) -> int:
    """Update quotes in the database with tags. Returns count actually updated."""
    count = 0
    for result in results:
        if db.update_tagged(
            conn,
            result.quote_id,
            keywords=result.keywords,
            categories=result.categories,
            batch_id=batch_id,
            author_type=result.author_type,
            religious_sentiment=result.religious_sentiment,
        ):
            count += 1
    return count
