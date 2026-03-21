"""Tests for the AI tagger response parsing."""

from wikiquotes_tagger.tagger import _parse_tag_response, _build_prompt
from wikiquotes_tagger.config import AppConfig


class TestParseTagResponse:
    def test_valid_json(self):
        response = '''[
            {"id": 1, "keywords": ["wisdom", "knowledge"], "category": "Philosophy"},
            {"id": 2, "keywords": ["love", "passion"], "category": "Romance"}
        ]'''
        results = _parse_tag_response(response, [100, 200])
        assert len(results) == 2
        assert results[0].quote_id == 100
        assert results[0].keywords == ["wisdom", "knowledge"]
        assert results[0].category == "Philosophy"
        assert results[1].quote_id == 200

    def test_markdown_fenced_json(self):
        response = '''```json
[{"id": 1, "keywords": ["test"], "category": "Test"}]
```'''
        results = _parse_tag_response(response, [42])
        assert len(results) == 1
        assert results[0].quote_id == 42

    def test_truncated_response(self):
        response = '''[
            {"id": 1, "keywords": ["wisdom"], "category": "Philosophy"},
            {"id": 2, "keywords": ["lo'''
        results = _parse_tag_response(response, [10, 20])
        # Should salvage the first complete object
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
        assert results[0].keywords == ["first"]  # Keeps first occurrence

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
        """json_mode may cause the model to wrap the array in an object."""
        response = '{"results": [{"id": 1, "keywords": ["test"], "category": "Test"}]}'
        results = _parse_tag_response(response, [99])
        assert len(results) == 1
        assert results[0].quote_id == 99


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
