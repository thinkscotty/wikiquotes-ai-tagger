# wikiquotes-ai-tagger

Standalone CLI tool that downloads an English [Wikiquote](https://en.wikiquote.org/) XML dump, extracts clean quotes, and uses an AI API to tag each quote with keywords, categories, author type classification, and religious sentiment analysis. Produces a SQLite database for use with Kibble Fetch.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An API key for an OpenAI-compatible AI provider (e.g., [Chutes](https://chutes.ai), [Together](https://together.ai), [Groq](https://groq.com))
- ~2 GB free disk space (198 MB compressed dump + 1.5 GB decompressed XML + database)

## Setup

```bash
git clone https://github.com/thinkscotty/wikiquotes-ai-tagger.git
cd wikiquotes-ai-tagger
uv sync
```

Set your API key as an environment variable. The variable name is configured in `config.toml` (default: `CHUTES_API_KEY`):

```bash
export CHUTES_API_KEY="your-key-here"
```

To persist across sessions, add the export line to your `~/.bashrc`, `~/.zshrc`, or equivalent.

## Usage

The tool runs a three-phase pipeline: **download → parse → tag**. Each phase is a separate CLI command. Run them in order.

### 1. Download the Wikiquote dump

```bash
uv run wikiquotes-tagger download
```

Downloads `enwikiquote-latest-pages-articles.xml.bz2` (~198 MB) from Wikimedia and decompresses it to XML (~1.5 GB). Skips download if the file already exists and matches the remote file size.

### 2. Parse quotes into SQLite

```bash
uv run wikiquotes-tagger parse
```

Stream-parses the XML dump and inserts quotes into `data/quotes.db`. Extracts quotes from:

- **Person pages** — quotes by a specific author (e.g., Albert Einstein, Mark Twain)
- **Literary work pages** — quotes from novels, plays, poems, and other published works (e.g., Hamlet, Pride and Prejudice, Moby-Dick)

Film/TV pages and theme pages (Love, War, etc.) are filtered out. Expects ~300,000 quotes from the current dump.

### 3. Tag quotes with AI

```bash
# Test with a small batch first
uv run wikiquotes-tagger tag --limit 20

# Tag a random sample for prompt quality testing (avoids author skew)
uv run wikiquotes-tagger tag --sample 1000

# Tag up to 500 quotes sequentially
uv run wikiquotes-tagger tag --limit 500

# Full run (can Ctrl+C and resume anytime)
uv run wikiquotes-tagger tag
```

Sends untagged quotes to an AI API in batches. For each quote, the AI produces:

- **Keywords**: 6–10 relevance-ordered keywords for topic matching
- **Categories**: 1–3 categories from the canonical list in `categories.txt`
- **Author type**: classification of the author field (`person`, `work`, `concept`, `religious_text`, `fictional`)
- **Religious sentiment** (optional): per-religion sentiment mapping for quotes about specific faiths

Progress is saved to the database after each batch — interrupt with Ctrl+C and resume at any time.

**Options:**

| Option | Description |
|--------|-------------|
| `--limit N` | Tag at most N quotes this run (useful for cost control and testing) |
| `--sample N` | Tag N quotes chosen at random from the full untagged pool. Avoids the author skew of sequential processing (the XML dump is alphabetical by page title, so `--limit 1000` would only cover authors starting with A–B). Already-tagged sample quotes are skipped by subsequent `tag` runs. |
| `--batch-size N` | Override the `batch_size` from config.toml for this run |
| `--debug` | Print raw API responses for each batch |

### Reset tags for re-tagging

```bash
# Reset all tagged and errored quotes back to 'parsed'
uv run wikiquotes-tagger reset --confirm
```

Clears all keywords, categories, author_type, and religious_sentiment from previously tagged quotes, setting their status back to `parsed`. Use this before re-running the tagger with an updated prompt or category list.

### Check progress

```bash
uv run wikiquotes-tagger stats
```

Shows total/parsed/tagged/errored counts, progress percentage, and top categories.

### Global options

All commands accept `--config PATH` to use a config file other than the default `config.toml`:

```bash
uv run wikiquotes-tagger --config my-config.toml tag --limit 100
```

## Categories

The file `categories.txt` in the project root defines the canonical list of categories the AI can assign. One category per line; lines starting with `#` are comments. The list is injected into the AI prompt at runtime via the `{{categories}}` placeholder in `config.toml`.

To modify the category list:

1. Edit `categories.txt` — add, remove, or rename categories
2. Run `uv run wikiquotes-tagger reset --confirm` to clear existing tags
3. Re-run `uv run wikiquotes-tagger tag` to re-tag with the updated list

The AI is instructed to choose only from this list and never invent categories. If a quote doesn't fit any category well, it uses the closest match. `Other` is a last-resort category for quotes that genuinely don't fit anywhere.

When adding categories, consider how users will search for quotes in Kibble. Categories should match natural search terms (e.g., "Courage", "Humor", "Christianity") rather than academic classifications.

## Configuration

Edit `config.toml` to change the AI provider, model, prompt, or tuning parameters.

### Switch AI provider

Change `base_url`, `model`, and `api_key_env` under `[api]`:

```toml
[api]
base_url = "https://api.together.xyz/v1"
model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
api_key_env = "TOGETHER_API_KEY"
```

Then set the corresponding environment variable (`export TOGETHER_API_KEY="..."`).

### Tuning parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `batch_size` | `20` | Quotes per API call. Higher = fewer calls but longer responses and higher truncation risk. |
| `timeout_seconds` | `120` | Per-request timeout in seconds. Increase for slow providers or large batches. |
| `delay_between_batches` | `5.0` | Seconds to wait between batches. Prevents rate limiting. |
| `max_retries` | `4` | Retry count for 429/5xx errors and consecutive failures before marking quotes as errored. |
| `json_mode` | `false` | Sends `response_format=json_object` to the API. Set to `true` if your model supports it for more reliable JSON. Can cause some models to return single objects instead of arrays. |
| `max_concurrent` | `5` | Reserved for future parallel batching. Currently unused. |

### Customize the prompt

Edit the `[prompts]` section in `config.toml`. Two placeholders are available:

- `{{categories}}` in the system prompt — replaced at runtime with the comma-separated contents of `categories.txt`
- `{{quotes}}` in the user prompt — replaced with the numbered quote list for each batch

```toml
[prompts]
system_prompt = """Your custom instructions here... Categories: {{categories}}"""
user_prompt_template = """Tag these quotes:\n\n{{quotes}}\n\nReturn JSON."""
```

## Output

The SQLite database (`data/quotes.db`) contains a single `quotes` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `text` | TEXT | Quote text (cleaned of wiki markup) |
| `author` | TEXT | Person the quote is attributed to (or work/concept title from Wikiquote) |
| `source_work` | TEXT | Book, speech, play title, etc. (null if unknown) |
| `source_confidence` | TEXT | `sourced` or `attributed` |
| `keywords` | TEXT | JSON array of lowercase keywords, ordered by relevance (null until tagged) |
| `category` | TEXT | JSON array of 1–3 category strings from `categories.txt` (null until tagged) |
| `author_type` | TEXT | `person`, `work`, `concept`, `religious_text`, or `fictional` (null until tagged) |
| `religious_sentiment` | TEXT | JSON object mapping religion names to sentiment, e.g. `{"christianity": "positive"}`. Null for non-religious quotes. |
| `status` | TEXT | `parsed`, `tagged`, or `error` |
| `batch_id` | INTEGER | Which AI batch processed this quote (null until tagged) |
| `created_at` | TEXT | Timestamp when the quote was inserted |

Query examples:

```sql
-- Browse tagged quotes
SELECT text, author, category, keywords FROM quotes WHERE status = 'tagged' LIMIT 10;

-- Find quotes by keyword
SELECT text, author FROM quotes WHERE keywords LIKE '%courage%';

-- Count quotes per category (categories are stored as JSON arrays)
SELECT json_each.value as cat, COUNT(*) as n
FROM quotes, json_each(category)
WHERE status = 'tagged'
GROUP BY cat ORDER BY n DESC;

-- Find quotes positive about Christianity
SELECT text, author FROM quotes
WHERE religious_sentiment IS NOT NULL
  AND json_extract(religious_sentiment, '$.christianity') = 'positive';

-- Count author types
SELECT author_type, COUNT(*) FROM quotes WHERE status = 'tagged' GROUP BY author_type;

-- Re-queue errored quotes for retry
UPDATE quotes SET status = 'parsed' WHERE status = 'error';
```

## Resume and error handling

The tagging phase is designed to be paused and resumed across multiple sessions. There is no external progress file — the SQLite database **is** the state:

- **Untagged quotes** have `status='parsed'` — the tagger queries for the next batch each time it runs.
- **Ctrl+C** finishes the current batch before exiting, so no work is lost.
- **API errors** are retried with exponential backoff. After multiple consecutive failures on the same batch, those quotes are marked `status='error'` and skipped so the rest of the corpus can proceed.
- **Errored quotes** can be re-queued with `uv run wikiquotes-tagger reset --confirm` or manually with `UPDATE quotes SET status = 'parsed' WHERE status = 'error';`.

## Development

```bash
uv sync --extra dev                    # Install dev dependencies
uv run --extra dev pytest -v           # Run tests
```
