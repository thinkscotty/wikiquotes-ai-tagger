# wikiquotes-ai-tagger

Standalone CLI tool that downloads an English [Wikiquote](https://en.wikiquote.org/) XML dump, extracts clean quotes, and uses an AI API to tag each quote with keywords and a category. Produces a SQLite database for use with Kibble Fetch.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An API key for an OpenAI-compatible AI provider (e.g., [Chutes](https://chutes.ai), [Together](https://together.ai), [Groq](https://groq.com))
- ~2 GB free disk space (198 MB compressed dump + 1.5 GB decompressed XML + database)

## Setup

```bash
git clone <repo-url>
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

Film/TV pages and theme pages (Love, War, etc.) are filtered out. Expects 30,000–60,000+ quotes depending on the dump.

### 3. Tag quotes with AI

```bash
# Test with a small batch first
uv run wikiquotes-tagger tag --limit 20

# Tag up to 500 quotes
uv run wikiquotes-tagger tag --limit 500

# Full run (can Ctrl+C and resume anytime)
uv run wikiquotes-tagger tag
```

Sends untagged quotes to an AI API in batches, receiving 6–10 relevance-ordered keywords and a freeform category per quote. Progress is saved to the database after each batch — interrupt with Ctrl+C and resume at any time.

**Options:**

| Option | Description |
|--------|-------------|
| `--limit N` | Tag at most N quotes this run (useful for cost control and testing) |
| `--batch-size N` | Override the `batch_size` from config.toml for this run |

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
| `json_mode` | `true` | Sends `response_format=json_object` to the API for more reliable JSON output. Set to `false` if your model doesn't support it. |
| `max_concurrent` | `5` | Reserved for future parallel batching. Currently unused. |

### Customize the prompt

Edit the `[prompts]` section. The `{{quotes}}` placeholder in `user_prompt_template` is replaced with the numbered quote list at runtime:

```toml
[prompts]
system_prompt = """Your custom instructions here..."""
user_prompt_template = """Tag these quotes:\n\n{{quotes}}\n\nReturn JSON."""
```

## Output

The SQLite database (`data/quotes.db`) contains a single `quotes` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `text` | TEXT | Quote text (cleaned of wiki markup) |
| `author` | TEXT | Person the quote is attributed to |
| `source_work` | TEXT | Book, speech, play title, etc. (null if unknown) |
| `source_confidence` | TEXT | `sourced` or `attributed` |
| `keywords` | TEXT | JSON array of lowercase keywords, ordered by relevance (null until tagged) |
| `category` | TEXT | Freeform category label (null until tagged) |
| `status` | TEXT | `parsed`, `tagged`, or `error` |
| `batch_id` | INTEGER | Which AI batch processed this quote (null until tagged) |
| `created_at` | TEXT | Timestamp when the quote was inserted |

Query examples:

```sql
-- Browse tagged quotes
SELECT text, author, keywords, category FROM quotes WHERE status = 'tagged' LIMIT 10;

-- Find quotes by keyword
SELECT text, author FROM quotes WHERE keywords LIKE '%courage%';

-- Count quotes per category
SELECT category, COUNT(*) as n FROM quotes WHERE status = 'tagged' GROUP BY category ORDER BY n DESC;

-- Re-queue errored quotes for retry
UPDATE quotes SET status = 'parsed' WHERE status = 'error';
```

## Resume and error handling

The tagging phase is designed to be paused and resumed across multiple sessions. There is no external progress file — the SQLite database **is** the state:

- **Untagged quotes** have `status='parsed'` — the tagger queries for the next batch each time it runs.
- **Ctrl+C** finishes the current batch before exiting, so no work is lost.
- **API errors** are retried with exponential backoff. After multiple consecutive failures on the same batch, those quotes are marked `status='error'` and skipped so the rest of the corpus can proceed.
- **Errored quotes** can be re-queued by running `UPDATE quotes SET status = 'parsed' WHERE status = 'error';` in the SQLite database.

## Development

```bash
uv sync --extra dev                    # Install dev dependencies
uv run --extra dev pytest -v           # Run tests (61 tests)
```
