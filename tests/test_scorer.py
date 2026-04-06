"""Tests for the quote quality scorer."""

import json
import sqlite3
import tempfile
from pathlib import Path

from wikiquotes_tagger.scorer import _parse_score_response, _build_scoring_prompt
from wikiquotes_tagger.config import ScoringPromptConfig, CalibrationExample, load_scoring_config
from wikiquotes_tagger import db


class TestParseScoreResponse:
    def test_valid_json(self):
        response = '[{"id": 1, "score": 8}, {"id": 2, "score": 3}]'
        results = _parse_score_response(response, [100, 200])
        assert len(results) == 2
        assert results[0].quote_id == 100
        assert results[0].score == 8
        assert results[1].quote_id == 200
        assert results[1].score == 3

    def test_markdown_fenced_json(self):
        response = '```json\n[{"id": 1, "score": 7}]\n```'
        results = _parse_score_response(response, [42])
        assert len(results) == 1
        assert results[0].score == 7

    def test_truncated_response(self):
        response = '[{"id": 1, "score": 9}, {"id": 2, "sco'
        results = _parse_score_response(response, [10, 20])
        assert len(results) == 1
        assert results[0].quote_id == 10
        assert results[0].score == 9

    def test_invalid_json(self):
        results = _parse_score_response("this is not json", [1, 2])
        assert len(results) == 0

    def test_no_array(self):
        results = _parse_score_response("just some text with no brackets", [1])
        assert len(results) == 0

    def test_score_clamped_high(self):
        response = '[{"id": 1, "score": 15}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 1
        assert results[0].score == 10

    def test_score_clamped_low(self):
        response = '[{"id": 1, "score": 0}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 1
        assert results[0].score == 1

    def test_score_negative_clamped(self):
        response = '[{"id": 1, "score": -3}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 1
        assert results[0].score == 1

    def test_fractional_score_rounded(self):
        response = '[{"id": 1, "score": 7.6}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 1
        assert results[0].score == 8

    def test_string_score_parsed(self):
        response = '[{"id": 1, "score": "6"}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 1
        assert results[0].score == 6

    def test_non_numeric_score_skipped(self):
        response = '[{"id": 1, "score": "high"}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 0

    def test_missing_score_skipped(self):
        response = '[{"id": 1}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 0

    def test_missing_id_skipped(self):
        response = '[{"score": 7}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 0

    def test_out_of_range_id(self):
        response = '[{"id": 99, "score": 7}]'
        results = _parse_score_response(response, [1, 2, 3])
        assert len(results) == 0

    def test_duplicate_ids_keeps_first(self):
        response = '[{"id": 1, "score": 8}, {"id": 1, "score": 3}]'
        results = _parse_score_response(response, [42])
        assert len(results) == 1
        assert results[0].score == 8

    def test_id_as_string(self):
        response = '[{"id": "1", "score": 5}]'
        results = _parse_score_response(response, [99])
        assert len(results) == 1
        assert results[0].quote_id == 99

    def test_mixed_valid_and_invalid(self):
        response = json.dumps([
            {"id": 1, "score": 7},
            {"id": 2, "score": "bad"},
            {"id": 3, "score": 4},
        ])
        results = _parse_score_response(response, [10, 20, 30])
        assert len(results) == 2
        assert results[0].quote_id == 10
        assert results[0].score == 7
        assert results[1].quote_id == 30
        assert results[1].score == 4

    def test_null_score_skipped(self):
        response = '[{"id": 1, "score": null}]'
        results = _parse_score_response(response, [1])
        assert len(results) == 0


class TestBuildScoringPrompt:
    def test_formats_quotes(self):
        config = ScoringPromptConfig(
            system_prompt="Rate quotes. {{calibration_examples}}",
            user_prompt_template="Rate:\n{{quotes}}",
        )
        quotes = [
            {"id": 1, "text": "To be or not to be.", "author": "Shakespeare"},
            {"id": 2, "text": "I think therefore I am.", "author": "Descartes"},
        ]
        system_msg, user_msg = _build_scoring_prompt(config, quotes)
        assert "1." in user_msg
        assert "Shakespeare" in user_msg
        assert "Descartes" in user_msg
        assert "{{quotes}}" not in user_msg

    def test_calibration_examples_injected(self):
        config = ScoringPromptConfig(
            system_prompt="Examples: {{calibration_examples}}",
            user_prompt_template="{{quotes}}",
            calibration=(
                CalibrationExample(
                    text="Fear itself",
                    author="FDR",
                    score=9,
                    reason="Iconic",
                ),
            ),
        )
        quotes = [{"id": 1, "text": "Test", "author": "Test"}]
        system_msg, _ = _build_scoring_prompt(config, quotes)
        assert "Fear itself" in system_msg
        assert "FDR" in system_msg
        assert "Score: 9" in system_msg
        assert "Iconic" in system_msg
        assert "{{calibration_examples}}" not in system_msg

    def test_no_calibration_examples(self):
        config = ScoringPromptConfig(
            system_prompt="Rate. {{calibration_examples}}",
            user_prompt_template="{{quotes}}",
        )
        quotes = [{"id": 1, "text": "Test", "author": "Test"}]
        system_msg, _ = _build_scoring_prompt(config, quotes)
        assert "No calibration examples provided" in system_msg

    def test_multiple_calibration_examples(self):
        config = ScoringPromptConfig(
            system_prompt="{{calibration_examples}}",
            user_prompt_template="{{quotes}}",
            calibration=(
                CalibrationExample(text="Quote A", author="Author A", score=9, reason="Great"),
                CalibrationExample(text="Quote B", author="Author B", score=2, reason="Poor"),
            ),
        )
        quotes = [{"id": 1, "text": "Test", "author": "Test"}]
        system_msg, _ = _build_scoring_prompt(config, quotes)
        assert "Quote A" in system_msg
        assert "Quote B" in system_msg
        assert "Score: 9" in system_msg
        assert "Score: 2" in system_msg


class TestLoadScoringConfig:
    def test_loads_valid_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[prompts]
system_prompt = "Rate quotes."
user_prompt_template = "{{quotes}}"

[[calibration]]
text = "Test quote"
author = "Test Author"
score = 7
reason = "Good quote"
""")
            f.flush()
            config = load_scoring_config(Path(f.name))
        assert config.system_prompt == "Rate quotes."
        assert config.user_prompt_template == "{{quotes}}"
        assert len(config.calibration) == 1
        assert config.calibration[0].text == "Test quote"
        assert config.calibration[0].score == 7

    def test_missing_file_raises(self):
        try:
            load_scoring_config(Path("/nonexistent/scoring_prompt.toml"))
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_empty_calibration(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[prompts]
system_prompt = "Rate."
user_prompt_template = "{{quotes}}"
""")
            f.flush()
            config = load_scoring_config(Path(f.name))
        assert len(config.calibration) == 0

    def test_skips_calibration_without_text(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[prompts]
system_prompt = "Rate."
user_prompt_template = "{{quotes}}"

[[calibration]]
author = "No text"
score = 5
reason = "Missing text field"
""")
            f.flush()
            config = load_scoring_config(Path(f.name))
        assert len(config.calibration) == 0


class TestDbScoringHelpers:
    """Test the database helper functions for scoring."""

    def _make_db(self) -> tuple[sqlite3.Connection, Path]:
        tmp = tempfile.mkdtemp()
        db_path = Path(tmp) / "test.db"
        db.init_db(db_path)
        conn = db.get_connection(db_path)
        return conn, db_path

    def _insert_tagged_quote(self, conn: sqlite3.Connection, quote_id: int, text: str) -> None:
        conn.execute(
            "INSERT INTO quotes (id, text, author, status) VALUES (?, ?, 'Test', 'tagged')",
            (quote_id, text),
        )
        conn.commit()

    def test_get_unscored_batch(self):
        conn, _ = self._make_db()
        self._insert_tagged_quote(conn, 1, "Quote A")
        self._insert_tagged_quote(conn, 2, "Quote B")
        conn.execute("UPDATE quotes SET quality = 7 WHERE id = 1")
        conn.commit()

        batch = db.get_unscored_batch(conn, 10)
        assert len(batch) == 1
        assert batch[0]["id"] == 2
        conn.close()

    def test_get_unscored_batch_respects_limit(self):
        conn, _ = self._make_db()
        for i in range(5):
            self._insert_tagged_quote(conn, i + 1, f"Quote {i}")

        batch = db.get_unscored_batch(conn, 2)
        assert len(batch) == 2
        conn.close()

    def test_update_quality(self):
        conn, _ = self._make_db()
        self._insert_tagged_quote(conn, 1, "Quote A")

        assert db.update_quality(conn, 1, 8) is True
        row = conn.execute("SELECT quality FROM quotes WHERE id = 1").fetchone()
        assert row[0] == 8
        conn.close()

    def test_update_quality_nonexistent(self):
        conn, _ = self._make_db()
        assert db.update_quality(conn, 999, 5) is False
        conn.close()

    def test_reset_scores(self):
        conn, _ = self._make_db()
        self._insert_tagged_quote(conn, 1, "Quote A")
        self._insert_tagged_quote(conn, 2, "Quote B")
        conn.execute("UPDATE quotes SET quality = 7 WHERE id = 1")
        conn.execute("UPDATE quotes SET quality = 3 WHERE id = 2")
        conn.commit()

        count = db.reset_scores(conn)
        assert count == 2
        row = conn.execute("SELECT quality FROM quotes WHERE id = 1").fetchone()
        assert row[0] is None
        conn.close()

    def test_get_scoring_stats(self):
        conn, _ = self._make_db()
        self._insert_tagged_quote(conn, 1, "Quote A")
        self._insert_tagged_quote(conn, 2, "Quote B")
        self._insert_tagged_quote(conn, 3, "Quote C")
        conn.execute("UPDATE quotes SET quality = 8 WHERE id = 1")
        conn.execute("UPDATE quotes SET quality = 4 WHERE id = 2")
        conn.commit()

        stats = db.get_scoring_stats(conn)
        assert stats["tagged"] == 3
        assert stats["scored"] == 2
        assert stats["unscored"] == 1
        assert stats["avg_quality"] == 6.0
        assert stats["distribution"] == {4: 1, 8: 1}
        conn.close()

    def test_get_scoring_stats_no_scores(self):
        conn, _ = self._make_db()
        self._insert_tagged_quote(conn, 1, "Quote A")

        stats = db.get_scoring_stats(conn)
        assert stats["scored"] == 0
        assert stats["avg_quality"] is None
        assert stats["distribution"] == {}
        conn.close()

    def test_get_random_unscored_ids(self):
        conn, _ = self._make_db()
        self._insert_tagged_quote(conn, 1, "Quote A")
        self._insert_tagged_quote(conn, 2, "Quote B")
        self._insert_tagged_quote(conn, 3, "Quote C")
        conn.execute("UPDATE quotes SET quality = 7 WHERE id = 1")
        conn.commit()

        ids = db.get_random_unscored_ids(conn, 10)
        assert set(ids) == {2, 3}
        conn.close()
