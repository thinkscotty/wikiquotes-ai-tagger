"""Tests for the AI tagger response parsing."""

import json
import tempfile
from pathlib import Path

from wikiquotes_tagger.tagger import _parse_tag_response, _build_prompt
from wikiquotes_tagger.config import AppConfig, PromptConfig, load_categories


class TestParseTagResponse:
    def test_valid_json_with_categories_array(self):
        response = '''[
            {"id": 1, "keywords": ["wisdom", "knowledge"], "categories": ["Philosophy", "Education"], "author_type": "person"},
            {"id": 2, "keywords": ["love", "passion"], "categories": ["Love"], "author_type": "person"}
        ]'''
        results = _parse_tag_response(response, [100, 200])
        assert len(results) == 2
        assert results[0].quote_id == 100
        assert results[0].keywords == ["wisdom", "knowledge"]
        assert results[0].categories == ["Philosophy", "Education"]
        assert results[0].author_type == "person"
        assert results[0].religious_sentiment is None
        assert results[1].quote_id == 200
        assert results[1].categories == ["Love"]

    def test_legacy_single_category_string(self):
        """Legacy 'category' string should be wrapped into a list."""
        response = '[{"id": 1, "keywords": ["test"], "category": "Philosophy"}]'
        results = _parse_tag_response(response, [1])
        assert len(results) == 1
        assert results[0].categories == ["Philosophy"]

    def test_categories_capped_at_three(self):
        response = json.dumps([{
            "id": 1, "keywords": ["test"],
            "categories": ["A", "B", "C", "D", "E"],
        }])
        results = _parse_tag_response(response, [1])
        assert len(results[0].categories) == 3

    def test_markdown_fenced_json(self):
        response = '''```json
[{"id": 1, "keywords": ["test"], "categories": ["Test"], "author_type": "person"}]
```'''
        results = _parse_tag_response(response, [42])
        assert len(results) == 1
        assert results[0].quote_id == 42

    def test_truncated_response(self):
        response = '''[
            {"id": 1, "keywords": ["wisdom"], "categories": ["Philosophy"], "author_type": "person"},
            {"id": 2, "keywords": ["lo'''
        results = _parse_tag_response(response, [10, 20])
        assert len(results) == 1
        assert results[0].quote_id == 10

    def test_invalid_json(self):
        results = _parse_tag_response("this is not json at all", [1, 2])
        assert len(results) == 0

    def test_no_array(self):
        results = _parse_tag_response('{"id": 1}', [1])
        assert len(results) == 0

    def test_id_as_string(self):
        response = '[{"id": "1", "keywords": ["test"], "category": "Test"}]'
        results = _parse_tag_response(response, [99])
        assert len(results) == 1
        assert results[0].quote_id == 99

    def test_keywords_as_comma_string(self):
        response = '[{"id": 1, "keywords": "wisdom, knowledge, truth", "category": "Philosophy"}]'
        results = _parse_tag_response(response, [50])
        assert len(results) == 1
        assert results[0].keywords == ["wisdom", "knowledge", "truth"]

    def test_skips_malformed_entries(self):
        response = '''[
            {"id": 1, "keywords": ["good"], "category": "OK"},
            {"id": 2, "keywords": [], "category": "Bad"},
            {"id": 3, "category": "Missing keywords"},
            {"id": 4, "keywords": ["ok"], "category": ""}
        ]'''
        results = _parse_tag_response(response, [10, 20, 30, 40])
        assert len(results) == 1
        assert results[0].quote_id == 10

    def test_out_of_range_id(self):
        response = '[{"id": 99, "keywords": ["test"], "category": "Test"}]'
        results = _parse_tag_response(response, [1, 2, 3])
        assert len(results) == 0

    def test_keywords_lowercased(self):
        response = '[{"id": 1, "keywords": ["Wisdom", "TRUTH"], "category": "Philosophy"}]'
        results = _parse_tag_response(response, [1])
        assert results[0].keywords == ["wisdom", "truth"]

    def test_deduplicates_ids(self):
        response = '''[
            {"id": 1, "keywords": ["first"], "category": "First"},
            {"id": 1, "keywords": ["second"], "category": "Second"}
        ]'''
        results = _parse_tag_response(response, [42])
        assert len(results) == 1
        assert results[0].keywords == ["first"]

    def test_filters_empty_keywords_after_strip(self):
        response = '[{"id": 1, "keywords": [" ", "wisdom", "  "], "category": "Philosophy"}]'
        results = _parse_tag_response(response, [1])
        assert len(results) == 1
        assert results[0].keywords == ["wisdom"]

    def test_rejects_all_whitespace_keywords(self):
        response = '[{"id": 1, "keywords": [" ", "  ", "\t"], "category": "Philosophy"}]'
        results = _parse_tag_response(response, [1])
        assert len(results) == 0

    def test_json_object_wrapping_array(self):
        response = '{"results": [{"id": 1, "keywords": ["test"], "category": "Test"}]}'
        results = _parse_tag_response(response, [99])
        assert len(results) == 1
        assert results[0].quote_id == 99

    def test_single_json_object(self):
        response = '{"id": 1, "keywords": ["wisdom", "knowledge"], "category": "Philosophy"}'
        results = _parse_tag_response(response, [42])
        assert len(results) == 1
        assert results[0].quote_id == 42
        assert results[0].keywords == ["wisdom", "knowledge"]

    # --- author_type tests ---

    def test_author_type_defaults_to_person(self):
        response = '[{"id": 1, "keywords": ["test"], "category": "Test"}]'
        results = _parse_tag_response(response, [1])
        assert results[0].author_type == "person"

    def test_author_type_valid_values(self):
        for at in ["person", "work", "concept", "religious_text", "fictional"]:
            response = json.dumps([{"id": 1, "keywords": ["test"], "category": "Test", "author_type": at}])
            results = _parse_tag_response(response, [1])
            assert results[0].author_type == at

    def test_author_type_invalid_defaults_to_person(self):
        response = '[{"id": 1, "keywords": ["test"], "category": "Test", "author_type": "alien"}]'
        results = _parse_tag_response(response, [1])
        assert results[0].author_type == "person"

    # --- religious_sentiment tests ---

    def test_religious_sentiment_omitted(self):
        response = '[{"id": 1, "keywords": ["test"], "category": "Faith"}]'
        results = _parse_tag_response(response, [1])
        assert results[0].religious_sentiment is None

    def test_religious_sentiment_valid(self):
        response = json.dumps([{
            "id": 1, "keywords": ["faith", "prayer"], "categories": ["Christianity"],
            "religious_sentiment": {"christianity": "positive"},
        }])
        results = _parse_tag_response(response, [1])
        parsed = json.loads(results[0].religious_sentiment)
        assert parsed == {"christianity": "positive"}

    def test_religious_sentiment_multiple_religions(self):
        response = json.dumps([{
            "id": 1, "keywords": ["faith"], "categories": ["Philosophy"],
            "religious_sentiment": {"christianity": "neutral", "islam": "critical"},
        }])
        results = _parse_tag_response(response, [1])
        parsed = json.loads(results[0].religious_sentiment)
        assert parsed["christianity"] == "neutral"
        assert parsed["islam"] == "critical"

    def test_religious_sentiment_invalid_values_filtered(self):
        response = json.dumps([{
            "id": 1, "keywords": ["faith"], "categories": ["Faith"],
            "religious_sentiment": {"christianity": "positive", "islam": "very_bad"},
        }])
        results = _parse_tag_response(response, [1])
        parsed = json.loads(results[0].religious_sentiment)
        assert "christianity" in parsed
        assert "islam" not in parsed

    def test_religious_sentiment_all_invalid_results_in_none(self):
        response = json.dumps([{
            "id": 1, "keywords": ["faith"], "categories": ["Faith"],
            "religious_sentiment": {"christianity": "very_positive"},
        }])
        results = _parse_tag_response(response, [1])
        assert results[0].religious_sentiment is None

    def test_religious_sentiment_non_dict_ignored(self):
        response = json.dumps([{
            "id": 1, "keywords": ["faith"], "categories": ["Faith"],
            "religious_sentiment": "positive",
        }])
        results = _parse_tag_response(response, [1])
        assert results[0].religious_sentiment is None


class TestBuildPrompt:
    def test_formats_quotes(self):
        config = AppConfig()
        quotes = [
            {"id": 1, "text": "To be or not to be.", "author": "Shakespeare", "source_work": None},
            {"id": 2, "text": "I think therefore I am.", "author": "Descartes", "source_work": None},
        ]
        system_msg, user_msg = _build_prompt(config, quotes)
        assert "1." in user_msg
        assert "Shakespeare" in user_msg
        assert "Descartes" in user_msg
        assert "{{quotes}}" not in user_msg
        assert len(system_msg) > 0

    def test_categories_injected_into_prompt(self):
        config = AppConfig(
            categories=("Courage", "Humor", "Philosophy"),
            prompts=PromptConfig(
                system_prompt="Categories: {{categories}}. Tag quotes.",
                user_prompt_template="{{quotes}}",
            ),
        )
        quotes = [{"id": 1, "text": "Test", "author": "Test"}]
        system_msg, _ = _build_prompt(config, quotes)
        assert "Courage, Humor, Philosophy" in system_msg
        assert "{{categories}}" not in system_msg


class TestLoadCategories:
    def test_reads_categories(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Comment\nPhilosophy\nHumor\n\n# Another comment\nScience\n")
            f.flush()
            categories = load_categories(Path(f.name))
        assert categories == ["Humor", "Philosophy", "Science"]

    def test_strips_whitespace(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  Courage  \n  Love\n")
            f.flush()
            categories = load_categories(Path(f.name))
        assert categories == ["Courage", "Love"]

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Only comments\n\n")
            f.flush()
            try:
                load_categories(Path(f.name))
                assert False, "Should have raised ValueError"
            except ValueError:
                pass

    def test_missing_file_raises(self):
        try:
            load_categories(Path("/nonexistent/categories.txt"))
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass
