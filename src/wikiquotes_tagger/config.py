"""TOML configuration loader."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DEFAULT_PATH = Path("config.toml")
CATEGORIES_DEFAULT_PATH = Path("categories.txt")


def load_categories(path: Path = CATEGORIES_DEFAULT_PATH) -> list[str]:
    """Load the canonical category list from a text file.

    Lines starting with # are comments. Blank lines are ignored.
    Returns a sorted list of category names.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Categories file not found: {path}\n"
            "Create a categories.txt with one category per line."
        )

    categories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                categories.append(line)

    if not categories:
        raise ValueError(f"Categories file is empty: {path}")

    return sorted(categories)


@dataclass(frozen=True)
class ApiConfig:
    base_url: str = "https://llm.chutes.ai/v1"
    model: str = "deepseek-ai/DeepSeek-V3"
    api_key_env: str = "CHUTES_API_KEY"
    batch_size: int = 20
    max_concurrent: int = 5
    timeout_seconds: int = 60
    delay_between_batches: float = 1.0
    max_retries: int = 3
    json_mode: bool = False

    @property
    def api_key(self) -> str:
        """Resolve the API key from the named environment variable."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"API key not found: set the {self.api_key_env} environment variable"
            )
        return key


@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str = (
        "You are a quote classification assistant. For each numbered quote below, provide:\n"
        "- keywords: 5-8 freeform keywords ordered by relevance (most relevant first). "
        "Keywords should capture the quote's themes, emotions, and subject matter.\n"
        "- category: a single freeform category label (e.g., \"Philosophy\", \"Humor\", "
        "\"Motivation\", \"Science\", \"Literature\", \"Politics\")\n\n"
        "Return ONLY a valid JSON array. Each element must have:\n"
        '- "id": the quote number from the input\n'
        '- "keywords": array of lowercase keyword strings\n'
        '- "category": a single category string\n\n'
        "Do not include any text outside the JSON array."
    )
    user_prompt_template: str = (
        "Tag the following quotes with keywords and a category:\n\n"
        "{{quotes}}\n\n"
        "Return a JSON array with one object per quote."
    )


@dataclass(frozen=True)
class CalibrationExample:
    text: str
    author: str
    score: int
    reason: str


@dataclass(frozen=True)
class ScoringPromptConfig:
    system_prompt: str = ""
    user_prompt_template: str = ""
    calibration: tuple[CalibrationExample, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    db_path: Path = field(default_factory=lambda: Path("data/quotes.db"))
    categories: tuple[str, ...] = ()


def load_config(path: Path = CONFIG_DEFAULT_PATH) -> AppConfig:
    """Load and validate config from a TOML file. Missing file returns defaults."""
    if not path.exists():
        return AppConfig()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    api_raw = raw.get("api", {})
    prompts_raw = raw.get("prompts", {})

    api = ApiConfig(
        base_url=api_raw.get("base_url", ApiConfig.base_url),
        model=api_raw.get("model", ApiConfig.model),
        api_key_env=api_raw.get("api_key_env", ApiConfig.api_key_env),
        batch_size=api_raw.get("batch_size", ApiConfig.batch_size),
        max_concurrent=api_raw.get("max_concurrent", ApiConfig.max_concurrent),
        timeout_seconds=api_raw.get("timeout_seconds", ApiConfig.timeout_seconds),
        delay_between_batches=api_raw.get("delay_between_batches", ApiConfig.delay_between_batches),
        max_retries=api_raw.get("max_retries", ApiConfig.max_retries),
        json_mode=api_raw.get("json_mode", ApiConfig.json_mode),
    )

    prompts = PromptConfig(
        system_prompt=prompts_raw.get("system_prompt", PromptConfig.system_prompt),
        user_prompt_template=prompts_raw.get("user_prompt_template", PromptConfig.user_prompt_template),
    )

    # Load categories from file next to config, falling back to project root
    categories_path = path.parent / "categories.txt"
    if not categories_path.exists():
        categories_path = CATEGORIES_DEFAULT_PATH
    try:
        categories = tuple(load_categories(categories_path))
    except (FileNotFoundError, ValueError):
        categories = ()

    return AppConfig(
        api=api,
        prompts=prompts,
        data_dir=Path(raw.get("data_dir", "data")),
        db_path=Path(raw.get("db_path", "data/quotes.db")),
        categories=categories,
    )


SCORING_PROMPT_DEFAULT_PATH = Path("scoring_prompt.toml")


def load_scoring_config(path: Path = SCORING_PROMPT_DEFAULT_PATH) -> ScoringPromptConfig:
    """Load scoring prompt configuration from a TOML file.

    Raises FileNotFoundError if the file doesn't exist — scoring config
    is required when running the score command, unlike main config.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Scoring prompt config not found: {path}\n"
            "Create a scoring_prompt.toml file (see README for format)."
        )

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    prompts_raw = raw.get("prompts", {})

    calibration_raw = raw.get("calibration", [])
    calibration = tuple(
        CalibrationExample(
            text=ex.get("text", ""),
            author=ex.get("author", ""),
            score=int(ex.get("score", 5)),
            reason=ex.get("reason", ""),
        )
        for ex in calibration_raw
        if ex.get("text")
    )

    return ScoringPromptConfig(
        system_prompt=prompts_raw.get("system_prompt", ""),
        user_prompt_template=prompts_raw.get("user_prompt_template", ""),
        calibration=calibration,
    )
