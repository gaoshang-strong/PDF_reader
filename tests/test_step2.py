"""Tests for pipeline/step2_normalize.py — all tests use a mock client."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.step2_normalize import (
    DEFAULT_MODEL,
    REQUIRED_HEADINGS,
    discover_article_ids,
    make_client,
    normalize_markdown,
    process_article,
    validate_normalized,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NORM_TEMPLATE = """\
# Text

{text}

# Images

{images}

# Captions

{captions}

# Tables

{tables}
"""


def _make_norm_response(text: str = "", images: str = "", captions: str = "", tables: str = "") -> str:
    return _NORM_TEMPLATE.format(text=text, images=images, captions=captions, tables=tables)


def _mock_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = response_text
    return client


def _setup_article(
    tmp_path: Path,
    article_id: str,
    raw_content: str,
    length_status: str = "PASS",
) -> Path:
    """Create the article_text.raw.md and raw_article_report.json for an article."""
    article_dir = tmp_path / "articles" / article_id
    article_dir.mkdir(parents=True)
    (article_dir / "article_text.raw.md").write_text(raw_content, encoding="utf-8")
    char_count = len(raw_content)
    report = {
        "article_id": article_id,
        "raw_char_count": char_count,
        "raw_word_count": len(raw_content.split()),
        "estimated_token_count": char_count // 4,
        "length_status": length_status,
    }
    (article_dir / "raw_article_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return article_dir


# ---------------------------------------------------------------------------
# make_client
# ---------------------------------------------------------------------------

def test_make_client_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    client = make_client()
    assert client.api_key == "test-key-123"


def test_make_client_uses_explicit_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = make_client(api_key="explicit-key")
    assert client.api_key == "explicit-key"


def test_make_client_uses_default_base_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    client = make_client()
    assert "deepseek.com" in str(client.base_url)


def test_make_client_uses_env_base_url(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.example.com/v1")
    client = make_client()
    assert "custom.example.com" in str(client.base_url)


def test_make_client_uses_explicit_base_url(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    client = make_client(api_key="k", base_url="https://override.example.com/v1")
    assert "override.example.com" in str(client.base_url)


# ---------------------------------------------------------------------------
# normalize_markdown
# ---------------------------------------------------------------------------

def test_normalize_markdown_calls_api(tmp_path):
    norm_text = _make_norm_response(text="Article body.", images="", captions="", tables="")
    client = _mock_client(norm_text)
    result = normalize_markdown("raw text", client, DEFAULT_MODEL)
    assert result == norm_text
    client.chat.completions.create.assert_called_once()


def test_normalize_markdown_passes_model(tmp_path):
    client = _mock_client(_make_norm_response())
    normalize_markdown("raw", client, "my-custom-model")
    call_kwargs = client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "my-custom-model" or call_kwargs.args[0] if call_kwargs.args else True
    # check via keyword or positional
    kwargs = client.chat.completions.create.call_args[1]
    assert kwargs.get("model") == "my-custom-model"


def test_normalize_markdown_sends_system_and_user_messages():
    from pipeline.step2_normalize import SYSTEM_PROMPT
    client = _mock_client(_make_norm_response())
    normalize_markdown("my raw md", client, DEFAULT_MODEL)
    kwargs = client.chat.completions.create.call_args[1]
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "my raw md"


# ---------------------------------------------------------------------------
# validate_normalized
# ---------------------------------------------------------------------------

def test_validate_pass_ratio():
    raw = "x" * 1000
    norm = _make_norm_response(text="x" * 1000)  # all headings present, ratio ≈ 1.0
    result = validate_normalized(raw, norm)
    assert result["status"] == "PASS"
    assert result["missing_headings"] == []


def test_validate_pass_lower_boundary():
    raw = "x" * 1000
    norm = "x" * 950  # ratio 0.95
    result = validate_normalized(raw, norm)
    assert result["status"] == "PASS"


def test_validate_pass_upper_boundary():
    raw = "x" * 1000
    norm = "x" * 1050  # ratio 1.05
    result = validate_normalized(raw, norm)
    assert result["status"] == "PASS"


def test_validate_warn_below():
    raw = "x" * 1000
    norm = "x" * 920  # ratio 0.92 → WARN
    result = validate_normalized(raw, norm)
    assert result["status"] == "WARN"


def test_validate_warn_above():
    raw = "x" * 1000
    norm = "x" * 1080  # ratio 1.08 → WARN
    result = validate_normalized(raw, norm)
    assert result["status"] == "WARN"


def test_validate_fail_below():
    raw = "x" * 1000
    norm = "x" * 800  # ratio 0.8 → FAIL
    result = validate_normalized(raw, norm)
    assert result["status"] == "FAIL"


def test_validate_fail_above():
    raw = "x" * 1000
    norm = "x" * 1200  # ratio 1.2 → FAIL
    result = validate_normalized(raw, norm)
    assert result["status"] == "FAIL"


def test_validate_missing_headings():
    raw = "some text"
    norm = "# Text\n\nstuff\n\n# Images\n\n"  # missing Captions and Tables
    result = validate_normalized(raw, norm)
    assert "# Captions" in result["missing_headings"]
    assert "# Tables" in result["missing_headings"]
    assert "# Text" not in result["missing_headings"]
    assert "# Images" not in result["missing_headings"]


def test_validate_all_headings_present():
    raw = "x" * 100
    norm = _make_norm_response(text="x" * 100)
    result = validate_normalized(raw, norm)
    assert result["missing_headings"] == []


def test_validate_empty_raw():
    result = validate_normalized("", "anything")
    assert result["ratio"] == 0.0


# ---------------------------------------------------------------------------
# process_article
# ---------------------------------------------------------------------------

RAW_CONTENT = "This is the article text. " * 50  # ~1300 chars


def test_process_article_pass_calls_llm(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "PASS")
    norm_response = _make_norm_response(text=RAW_CONTENT)
    client = _mock_client(norm_response)
    result = process_article("pdf_abc", tmp_path, client)
    assert result["llm_called"] is True
    client.chat.completions.create.assert_called_once()


def test_process_article_warn_calls_llm(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "WARN")
    norm_response = _make_norm_response(text=RAW_CONTENT)
    client = _mock_client(norm_response)
    result = process_article("pdf_abc", tmp_path, client)
    assert result["llm_called"] is True


def test_process_article_fail_skips_llm_copies_raw(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "FAIL")
    client = _mock_client("should not be used")
    result = process_article("pdf_abc", tmp_path, client)
    assert result["llm_called"] is False
    client.chat.completions.create.assert_not_called()
    norm_path = tmp_path / "articles" / "pdf_abc" / "article_text.normalized.md"
    assert norm_path.read_text(encoding="utf-8") == RAW_CONTENT


def test_process_article_writes_normalized_md(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "PASS")
    norm_response = _make_norm_response(text=RAW_CONTENT)
    client = _mock_client(norm_response)
    process_article("pdf_abc", tmp_path, client)
    norm_path = tmp_path / "articles" / "pdf_abc" / "article_text.normalized.md"
    assert norm_path.exists()
    assert norm_path.read_text(encoding="utf-8") == norm_response


def test_process_article_skips_existing_by_default(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "PASS")
    norm_path = tmp_path / "articles" / "pdf_abc" / "article_text.normalized.md"
    norm_path.write_text("sentinel", encoding="utf-8")
    client = _mock_client(_make_norm_response())
    result = process_article("pdf_abc", tmp_path, client)
    assert result["skipped"] is True
    client.chat.completions.create.assert_not_called()
    assert norm_path.read_text() == "sentinel"


def test_process_article_overwrite_replaces_existing(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "PASS")
    norm_path = tmp_path / "articles" / "pdf_abc" / "article_text.normalized.md"
    norm_path.write_text("old content", encoding="utf-8")
    norm_response = _make_norm_response(text=RAW_CONTENT)
    client = _mock_client(norm_response)
    result = process_article("pdf_abc", tmp_path, client, overwrite=True)
    assert result["skipped"] is False
    assert result["llm_called"] is True
    assert norm_path.read_text(encoding="utf-8") == norm_response


def test_process_article_returns_validation_fields(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "PASS")
    norm_response = _make_norm_response(text=RAW_CONTENT)
    client = _mock_client(norm_response)
    result = process_article("pdf_abc", tmp_path, client)
    assert "ratio" in result
    assert "status" in result
    assert "missing_headings" in result


def test_process_article_uses_given_model(tmp_path):
    _setup_article(tmp_path, "pdf_abc", RAW_CONTENT, "PASS")
    client = _mock_client(_make_norm_response(text=RAW_CONTENT))
    process_article("pdf_abc", tmp_path, client, model="custom-model")
    kwargs = client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "custom-model"


# ---------------------------------------------------------------------------
# discover_article_ids
# ---------------------------------------------------------------------------

def test_discover_article_ids_from_articles_dir(tmp_path):
    for aid in ("pdf_bbb", "pdf_aaa", "pdf_ccc"):
        (tmp_path / "articles" / aid).mkdir(parents=True)
    ids = discover_article_ids(tmp_path)
    assert ids == ["pdf_aaa", "pdf_bbb", "pdf_ccc"]


def test_discover_article_ids_empty(tmp_path):
    (tmp_path / "articles").mkdir()
    assert discover_article_ids(tmp_path) == []


def test_discover_article_ids_missing_dir(tmp_path):
    assert discover_article_ids(tmp_path) == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_all_articles(tmp_path):
    from pipeline.step2_normalize import main as step2_main

    for aid in ("pdf_aaa", "pdf_bbb"):
        _setup_article(tmp_path, aid, RAW_CONTENT, "PASS")
    norm_response = _make_norm_response(text=RAW_CONTENT)
    mock_client = _mock_client(norm_response)

    with patch("pipeline.step2_normalize.make_client", return_value=mock_client):
        sys.argv = ["step2", "--data-root", str(tmp_path)]
        step2_main()

    for aid in ("pdf_aaa", "pdf_bbb"):
        assert (tmp_path / "articles" / aid / "article_text.normalized.md").exists()


def test_cli_single_article(tmp_path):
    from pipeline.step2_normalize import main as step2_main

    for aid in ("pdf_aaa", "pdf_bbb"):
        _setup_article(tmp_path, aid, RAW_CONTENT, "PASS")
    mock_client = _mock_client(_make_norm_response(text=RAW_CONTENT))

    with patch("pipeline.step2_normalize.make_client", return_value=mock_client):
        sys.argv = ["step2", "--data-root", str(tmp_path), "--article-id", "pdf_aaa"]
        step2_main()

    assert (tmp_path / "articles" / "pdf_aaa" / "article_text.normalized.md").exists()
    assert not (tmp_path / "articles" / "pdf_bbb" / "article_text.normalized.md").exists()


def test_cli_skip_existing(tmp_path):
    from pipeline.step2_normalize import main as step2_main

    _setup_article(tmp_path, "pdf_aaa", RAW_CONTENT, "PASS")
    norm_path = tmp_path / "articles" / "pdf_aaa" / "article_text.normalized.md"
    norm_path.write_text("existing", encoding="utf-8")
    mock_client = _mock_client(_make_norm_response())

    with patch("pipeline.step2_normalize.make_client", return_value=mock_client):
        sys.argv = [
            "step2",
            "--data-root",
            str(tmp_path),
            "--article-id",
            "pdf_aaa",
            "--skip-existing",
        ]
        step2_main()

    assert norm_path.read_text() == "existing"
    mock_client.chat.completions.create.assert_not_called()


def test_cli_overwrite(tmp_path):
    from pipeline.step2_normalize import main as step2_main

    _setup_article(tmp_path, "pdf_aaa", RAW_CONTENT, "PASS")
    norm_path = tmp_path / "articles" / "pdf_aaa" / "article_text.normalized.md"
    norm_path.write_text("old", encoding="utf-8")
    fresh = _make_norm_response(text=RAW_CONTENT)
    mock_client = _mock_client(fresh)

    with patch("pipeline.step2_normalize.make_client", return_value=mock_client):
        sys.argv = [
            "step2",
            "--data-root",
            str(tmp_path),
            "--article-id",
            "pdf_aaa",
            "--overwrite",
        ]
        step2_main()

    assert norm_path.read_text(encoding="utf-8") == fresh


def test_cli_custom_model(tmp_path):
    from pipeline.step2_normalize import main as step2_main

    _setup_article(tmp_path, "pdf_aaa", RAW_CONTENT, "PASS")
    mock_client = _mock_client(_make_norm_response(text=RAW_CONTENT))

    with patch("pipeline.step2_normalize.make_client", return_value=mock_client):
        sys.argv = [
            "step2",
            "--data-root",
            str(tmp_path),
            "--article-id",
            "pdf_aaa",
            "--model",
            "deepseek-special",
        ]
        step2_main()

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "deepseek-special"


def test_cli_fail_article_copies_raw(tmp_path):
    from pipeline.step2_normalize import main as step2_main

    raw = "raw content that is very long " * 10
    _setup_article(tmp_path, "pdf_aaa", raw, "FAIL")
    mock_client = _mock_client("should not appear")

    with patch("pipeline.step2_normalize.make_client", return_value=mock_client):
        sys.argv = ["step2", "--data-root", str(tmp_path), "--article-id", "pdf_aaa"]
        step2_main()

    norm_path = tmp_path / "articles" / "pdf_aaa" / "article_text.normalized.md"
    assert norm_path.read_text(encoding="utf-8") == raw
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt content — LaTeX normalization instructions
# ---------------------------------------------------------------------------

def _get_system_prompt() -> str:
    from pipeline.step2_normalize import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def test_prompt_includes_latex_conversion_instruction():
    prompt = _get_system_prompt()
    assert "Convert" in prompt or "convert" in prompt
    # Must explicitly mention text-like LaTeX → plain text
    assert "text-like" in prompt or "text-like LaTeX" in prompt or "mathsf" in prompt or "mathtt" in prompt


def test_prompt_includes_latex_conversion_examples():
    prompt = _get_system_prompt()
    # At least one concrete gene/protein example must be present
    assert "CD8" in prompt or "CD4" in prompt or "TOX" in prompt


def test_prompt_includes_real_formula_preservation():
    prompt = _get_system_prompt()
    # Must explicitly say to keep real/actual formulas as LaTeX
    lower = prompt.lower()
    assert "preserve" in lower or "keep" in lower
    # Must mention at least two formula types
    formula_terms = ["fraction", "integral", "sum", "equation", "statistical", "formula"]
    matched = [t for t in formula_terms if t in lower]
    assert len(matched) >= 2, f"Expected ≥2 formula types in prompt, got: {matched}"


def test_prompt_includes_latex_uncertainty_fallback():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    # Must say to keep original LaTeX when uncertain
    assert "uncertain" in lower or "unsure" in lower
    assert "keep" in lower or "preserve" in lower or "unchanged" in lower


def test_prompt_latex_uncertainty_refers_to_original():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    # The uncertainty fallback should say to keep the LaTeX unchanged
    assert "unchanged" in lower or "original" in lower


# ---------------------------------------------------------------------------
# Prompt content — interrupted sentence repair instructions
# ---------------------------------------------------------------------------

def test_prompt_includes_sentence_repair_instruction():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    assert "interrupt" in lower or "repair" in lower or "continuous" in lower


def test_prompt_sentence_repair_mentions_image_as_interrupter():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    assert "image" in lower


def test_prompt_sentence_repair_mentions_non_image_interrupters():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    # Must acknowledge that non-image blocks can also interrupt sentences
    non_image_terms = ["caption", "table", "formula", "artifact", "layout"]
    matched = [t for t in non_image_terms if t in lower]
    assert len(matched) >= 2, f"Expected ≥2 non-image interrupter types, got: {matched}"


def test_prompt_sentence_repair_includes_example():
    prompt = _get_system_prompt()
    # The prompt should contain a concrete before/after example showing sentence repair
    assert "TME" in prompt or "predominance" in prompt or "Example" in prompt or "example" in prompt


def test_prompt_sentence_repair_display_formula_stays_in_text():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    # Must instruct that a display formula interrupting a sentence stays in Text
    assert "formula" in lower or "display" in lower
    assert "text" in lower  # keep it in Text section


def test_prompt_sentence_repair_uncertainty_fallback():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    # Must say to keep original separation when uncertain about joining
    assert "uncertain" in lower or "unsure" in lower
    # And say keep/preserve/separate
    assert any(w in lower for w in ("keep", "separate", "original"))


# ---------------------------------------------------------------------------
# Prompt content — general structure checks
# ---------------------------------------------------------------------------

def test_prompt_has_all_four_section_names():
    prompt = _get_system_prompt()
    for heading in REQUIRED_HEADINGS:
        assert heading in prompt, f"Prompt missing section heading: {heading}"


def test_prompt_instructs_no_summarization():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    assert "summarize" in lower or "shorten" in lower


def test_prompt_instructs_length_preservation():
    prompt = _get_system_prompt()
    lower = prompt.lower()
    assert "length" in lower or "close to" in lower
