"""Tests for pipeline/step3_structure_normalized.py

All tests use synthetic markdown strings — no LLM, no external API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.step3_structure_normalized import (
    parse_normalized_md,
    process_article,
    discover_article_ids,
    _build_line_offsets,
    _token_count,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTICLE_ID = "pdf_test000"


def blocks_of_type(blocks, btype):
    return [b for b in blocks if b["block_type"] == btype]


def _setup_article(tmp_path: Path, article_id: str, norm_content: str) -> Path:
    article_dir = tmp_path / "articles" / article_id
    article_dir.mkdir(parents=True)
    (article_dir / "article_text.normalized.md").write_text(norm_content, encoding="utf-8")
    return article_dir


# ---------------------------------------------------------------------------
# Line offset utility
# ---------------------------------------------------------------------------

def test_line_offsets_simple():
    text = "line0\nline1\nline2"
    offsets = _build_line_offsets(text)
    assert offsets[0] == 0
    assert offsets[1] == 6   # len("line0\n")
    assert offsets[2] == 12  # len("line0\nline1\n")


def test_line_offsets_empty():
    assert _build_line_offsets("") == []


def test_line_offsets_single_line():
    offsets = _build_line_offsets("hello")
    assert offsets == [0]


# ---------------------------------------------------------------------------
# Heading detection and section_path
# ---------------------------------------------------------------------------

def test_heading_single_level():
    md = "# Text\n\n# Introduction\n\nSome text.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    headings = blocks_of_type(blocks, "heading")
    heading_names = [h["text"] for h in headings]
    assert "Text" in heading_names
    assert "Introduction" in heading_names


def test_heading_level_stored():
    md = "# H1\n\n## H2\n\n### H3\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    by_text = {b["text"]: b for b in blocks}
    assert by_text["H1"]["heading_level"] == 1
    assert by_text["H2"]["heading_level"] == 2
    assert by_text["H3"]["heading_level"] == 3


def test_section_path_level1():
    md = "# Text\n\nParagraph here.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    assert para["section_path"] == ["Text"]


def test_section_path_level2():
    md = "# Text\n\n## Methods\n\nMethodology here.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    assert para["section_path"] == ["Text", "Methods"]


def test_section_path_level3():
    md = "# Text\n\n## Methods\n\n### Statistics\n\nStats paragraph.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    assert para["section_path"] == ["Text", "Methods", "Statistics"]


def test_section_path_resets_on_same_level():
    md = "# Text\n\n## Results\n\nResult text.\n\n## Discussion\n\nDiscussion text.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    paras = blocks_of_type(blocks, "paragraph")
    assert paras[0]["section_path"] == ["Text", "Results"]
    assert paras[1]["section_path"] == ["Text", "Discussion"]


def test_heading_heading_level_none_for_non_headings():
    md = "# Text\n\nJust a paragraph.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    assert para["heading_level"] is None


# ---------------------------------------------------------------------------
# Paragraph merging
# ---------------------------------------------------------------------------

def test_paragraph_merges_consecutive_lines():
    md = "# Text\n\nLine one.\nLine two.\nLine three.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    paras = blocks_of_type(blocks, "paragraph")
    assert len(paras) == 1
    assert "Line one." in paras[0]["text"]
    assert "Line two." in paras[0]["text"]
    assert "Line three." in paras[0]["text"]


def test_blank_line_creates_paragraph_boundary():
    md = "# Text\n\nFirst paragraph.\n\nSecond paragraph.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    paras = blocks_of_type(blocks, "paragraph")
    assert len(paras) == 2
    assert "First" in paras[0]["text"]
    assert "Second" in paras[1]["text"]


def test_paragraph_text_is_stripped():
    md = "# Text\n\n  Leading spaces.  \n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    paras = blocks_of_type(blocks, "paragraph")
    assert paras[0]["text"] == "Leading spaces."


def test_multiple_blank_lines_still_one_boundary():
    md = "# Text\n\nFirst.\n\n\n\nSecond.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    paras = blocks_of_type(blocks, "paragraph")
    assert len(paras) == 2


# ---------------------------------------------------------------------------
# Figure caption detection
# ---------------------------------------------------------------------------

def test_figure_caption_detected():
    md = "# Captions\n\nFigure 1. This is the first figure caption.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    caps = blocks_of_type(blocks, "figure_caption")
    assert len(caps) == 1
    assert caps[0]["text"].startswith("Figure 1.")


def test_figure_caption_abbreviated():
    md = "# Captions\n\nFig. 2. Another caption.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    caps = blocks_of_type(blocks, "figure_caption")
    assert len(caps) == 1
    assert "Fig. 2." in caps[0]["text"]


def test_figure_caption_multi_line():
    md = "# Captions\n\nFigure 3. First line of caption.\nContinued caption text.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    caps = blocks_of_type(blocks, "figure_caption")
    assert len(caps) == 1
    assert "Continued caption text." in caps[0]["text"]


def test_figure_caption_case_insensitive():
    md = "# Captions\n\nfigure 4. Lowercase figure.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    caps = blocks_of_type(blocks, "figure_caption")
    assert len(caps) == 1


def test_multiple_figure_captions():
    md = "# Captions\n\nFigure 1. First.\n\nFigure 2. Second.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    caps = blocks_of_type(blocks, "figure_caption")
    assert len(caps) == 2


# ---------------------------------------------------------------------------
# Table caption detection
# ---------------------------------------------------------------------------

def test_table_caption_detected():
    md = "# Captions\n\nTable 1. Summary of results.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    tcaps = blocks_of_type(blocks, "table_caption")
    assert len(tcaps) == 1
    assert "Table 1." in tcaps[0]["text"]


def test_table_caption_multi_line():
    md = "# Captions\n\nTable 2. Experimental conditions.\nWith continued description.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    tcaps = blocks_of_type(blocks, "table_caption")
    assert len(tcaps) == 1
    assert "Experimental conditions." in tcaps[0]["text"]
    assert "With continued description." in tcaps[0]["text"]


def test_table_and_figure_captions_coexist():
    md = "# Captions\n\nFigure 1. Figure cap.\n\nTable 1. Table cap.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    assert len(blocks_of_type(blocks, "figure_caption")) == 1
    assert len(blocks_of_type(blocks, "table_caption")) == 1


# ---------------------------------------------------------------------------
# References splitting
# ---------------------------------------------------------------------------

def test_references_heading_activates_reference_mode():
    md = "# REFERENCES\n\n1. Author A. Paper title. Journal 2020.\n2. Author B. Another paper. 2021.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    assert len(refs) == 2


def test_references_heading_case_insensitive():
    md = "# References\n\n1. Author A (2020). Title. J. 1.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    assert len(refs) == 1


def test_reference_text_preserved():
    ref_text = "1. Smith, J. et al. Some study. Nature 500, 100–110 (2020)."
    md = f"# REFERENCES\n\n{ref_text}\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    assert len(refs) == 1
    assert "Smith, J." in refs[0]["text"]
    assert "Nature 500" in refs[0]["text"]


def test_references_split_ten_items():
    lines = [f"{n}. Author {n}. Title {n}. J. {n} (2020)." for n in range(1, 11)]
    md = "# REFERENCES\n\n" + "\n".join(lines) + "\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    assert len(refs) == 10


def test_reference_multi_line_entry():
    md = (
        "# REFERENCES\n\n"
        "1. Author A. Very long title that continues\n"
        "   on the next line. Journal 2020.\n"
        "2. Author B. Short title. 2021.\n"
    )
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    assert len(refs) == 2
    assert "continues" in refs[0]["text"]


# ---------------------------------------------------------------------------
# Back matter classification
# ---------------------------------------------------------------------------

def test_acknowledgments_heading_is_back_matter():
    md = "# ACKNOWLEDGMENTS\n\nWe thank the funding agencies.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    bm = [b for b in blocks if b["block_type"] == "back_matter"]
    assert len(bm) >= 1


def test_author_contributions_heading():
    md = "# AUTHOR CONTRIBUTIONS\n\nA.B. designed the study.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    bm = [b for b in blocks if b["block_type"] == "back_matter"]
    assert any("designed the study" in b["text"] for b in bm)


def test_declaration_of_interests():
    md = "# DECLARATION OF INTERESTS\n\nThe authors declare no conflicts.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    bm = [b for b in blocks if b["block_type"] == "back_matter"]
    assert any("no conflicts" in b["text"] for b in bm)


def test_supplemental_information():
    md = "# SUPPLEMENTAL INFORMATION\n\nSupplemental data available online.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    bm = [b for b in blocks if b["block_type"] == "back_matter"]
    assert len(bm) >= 1


def test_data_and_code_availability():
    md = "# DATA AND CODE AVAILABILITY\n\nData deposited in GEO.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    bm = [b for b in blocks if b["block_type"] == "back_matter"]
    assert any("GEO" in b["text"] for b in bm)


# ---------------------------------------------------------------------------
# char_start / char_end span correctness
# ---------------------------------------------------------------------------

def test_char_span_heading():
    md = "# Introduction\n\nSome text.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    heading = [b for b in blocks if b["text"] == "Introduction"][0]
    span = md[heading["char_start"]: heading["char_end"]]
    assert "Introduction" in span


def test_char_span_paragraph():
    md = "# Text\n\nHello world paragraph.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    span = md[para["char_start"]: para["char_end"]]
    assert "Hello world paragraph." in span


def test_char_span_reference():
    md = "# REFERENCES\n\n1. Smith J. Title. 2020.\n2. Jones A. Other. 2021.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    for ref in refs:
        span = md[ref["char_start"]: ref["char_end"]]
        assert ref["text"].split()[0] in span  # first word appears in span


def test_char_span_figure_caption():
    md = "# Captions\n\nFigure 1. The caption text.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    cap = blocks_of_type(blocks, "figure_caption")[0]
    span = md[cap["char_start"]: cap["char_end"]]
    assert "Figure 1." in span


def test_char_start_less_than_char_end():
    md = "# Text\n\nParagraph with some content.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    for b in blocks:
        assert b["char_start"] <= b["char_end"], (
            f"block {b['block_id']} has char_start > char_end"
        )


def test_char_span_non_overlapping_and_ordered():
    """Blocks should be ordered and their spans should not go backwards."""
    md = "# Text\n\nFirst para.\n\n## Methods\n\nSecond para.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    for j in range(1, len(blocks)):
        assert blocks[j]["char_start"] >= blocks[j - 1]["char_start"], (
            f"Block {j} starts before block {j-1}"
        )


def test_char_span_covers_actual_text():
    md = "# Text\n\nThe quick brown fox.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    span = md[para["char_start"]: para["char_end"]]
    assert "quick brown fox" in span


# ---------------------------------------------------------------------------
# No text rewriting
# ---------------------------------------------------------------------------

def test_paragraph_text_not_rewritten():
    original = "Cancer cells degrade the nerve fibre myelin sheets."
    md = f"# Text\n\n{original}\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    assert para["text"] == original


def test_reference_text_not_rewritten():
    ref = "1. Liebig, C. et al. Perineural invasion in cancer. Cancer 115, 3379 (2009)."
    md = f"# REFERENCES\n\n{ref}\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    assert refs[0]["text"] == ref.strip()


def test_special_characters_preserved():
    original = "p = 0.001; β = 0.5; CD8⁺ T cells (n = 33)."
    md = f"# Text\n\n{original}\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    assert para["text"] == original


def test_latex_not_altered():
    original = r"The ratio $\frac{a}{b}$ was significant ($P < 0.01$)."
    md = f"# Text\n\n{original}\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    para = blocks_of_type(blocks, "paragraph")[0]
    assert para["text"] == original


# ---------------------------------------------------------------------------
# Block schema completeness
# ---------------------------------------------------------------------------

def test_all_blocks_have_required_fields():
    md = "# Text\n\nParagraph.\n\n# REFERENCES\n\n1. Ref one. 2020.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    required = {"block_id", "article_id", "block_index", "block_type",
                 "section_path", "heading_level", "text", "char_start",
                 "char_end", "token_count"}
    for b in blocks:
        assert required.issubset(b.keys()), f"Block missing fields: {required - b.keys()}"


def test_block_index_sequential():
    md = "# Text\n\nA.\n\nB.\n\nC.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    for i, b in enumerate(blocks):
        assert b["block_index"] == i


def test_article_id_in_all_blocks():
    md = "# Text\n\nContent here.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    for b in blocks:
        assert b["article_id"] == ARTICLE_ID


def test_block_id_contains_article_id():
    md = "# Text\n\nContent.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    for b in blocks:
        assert b["block_id"].startswith(ARTICLE_ID)


def test_section_path_is_list():
    md = "# Text\n\nContent.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    for b in blocks:
        assert isinstance(b["section_path"], list)


def test_token_count_positive():
    md = "# Text\n\nSome sentence here.\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    for b in blocks:
        assert b["token_count"] >= 1


# ---------------------------------------------------------------------------
# process_article — integration test
# ---------------------------------------------------------------------------

def test_process_article_creates_jsonl(tmp_path):
    md = "# Text\n\nParagraph.\n\n# REFERENCES\n\n1. Author. Title. 2020.\n"
    _setup_article(tmp_path, ARTICLE_ID, md)
    result = process_article(ARTICLE_ID, tmp_path)
    assert not result.get("skipped")
    jsonl_path = tmp_path / "index" / ARTICLE_ID / "structured_blocks.jsonl"
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == result["block_count"]


def test_process_article_creates_manifest(tmp_path):
    md = "# Text\n\nParagraph.\n"
    _setup_article(tmp_path, ARTICLE_ID, md)
    process_article(ARTICLE_ID, tmp_path)
    manifest_path = tmp_path / "index" / ARTICLE_ID / "structure_manifest.json"
    assert manifest_path.exists()
    m = json.loads(manifest_path.read_text())
    assert m["article_id"] == ARTICLE_ID
    assert "block_count" in m
    assert "block_type_distribution" in m


def test_process_article_skips_existing_without_overwrite(tmp_path):
    md = "# Text\n\nParagraph.\n"
    _setup_article(tmp_path, ARTICLE_ID, md)
    process_article(ARTICLE_ID, tmp_path)
    # Second call without overwrite
    result = process_article(ARTICLE_ID, tmp_path, overwrite=False)
    assert result.get("skipped")


def test_process_article_overwrites_with_flag(tmp_path):
    md = "# Text\n\nParagraph.\n"
    _setup_article(tmp_path, ARTICLE_ID, md)
    process_article(ARTICLE_ID, tmp_path)
    result = process_article(ARTICLE_ID, tmp_path, overwrite=True)
    assert not result.get("skipped")


def test_process_article_missing_normalized_md(tmp_path):
    # No normalized.md exists
    (tmp_path / "articles" / ARTICLE_ID).mkdir(parents=True)
    result = process_article(ARTICLE_ID, tmp_path)
    assert result.get("skipped")


def test_jsonl_lines_are_valid_json(tmp_path):
    md = "# Text\n\nParagraph one.\n\n## Methods\n\nParagraph two.\n"
    _setup_article(tmp_path, ARTICLE_ID, md)
    process_article(ARTICLE_ID, tmp_path)
    jsonl_path = tmp_path / "index" / ARTICLE_ID / "structured_blocks.jsonl"
    for line in jsonl_path.read_text().strip().split("\n"):
        obj = json.loads(line)
        assert isinstance(obj, dict)


# ---------------------------------------------------------------------------
# discover_article_ids
# ---------------------------------------------------------------------------

def test_discover_finds_articles_with_normalized_md(tmp_path):
    for aid in ("pdf_aaa", "pdf_bbb"):
        _setup_article(tmp_path, aid, "# Text\n\nContent.\n")
    ids = discover_article_ids(tmp_path)
    assert "pdf_aaa" in ids
    assert "pdf_bbb" in ids


def test_discover_skips_articles_without_normalized_md(tmp_path):
    (tmp_path / "articles" / "pdf_nomatch").mkdir(parents=True)
    ids = discover_article_ids(tmp_path)
    assert "pdf_nomatch" not in ids


def test_discover_returns_sorted(tmp_path):
    for aid in ("pdf_ccc", "pdf_aaa", "pdf_bbb"):
        _setup_article(tmp_path, aid, "# Text\n\nContent.\n")
    ids = discover_article_ids(tmp_path)
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_normalized_md():
    blocks = parse_normalized_md("", ARTICLE_ID)
    assert blocks == []


def test_only_headings_no_content():
    md = "# Text\n\n## Methods\n\n## Results\n"
    blocks = parse_normalized_md(md, ARTICLE_ID)
    assert all(b["block_type"] in ("heading",) for b in blocks)


def test_references_do_not_spill_into_next_section():
    md = (
        "# REFERENCES\n\n"
        "1. Ref one. 2020.\n"
        "2. Ref two. 2021.\n\n"
        "# Methods\n\n"
        "Methods paragraph.\n"
    )
    blocks = parse_normalized_md(md, ARTICLE_ID)
    refs = blocks_of_type(blocks, "reference")
    paras = blocks_of_type(blocks, "paragraph")
    assert len(refs) == 2
    assert len(paras) == 1
    assert "Methods paragraph." in paras[0]["text"]


def test_back_matter_does_not_affect_earlier_paragraphs():
    md = (
        "# Text\n\n"
        "Normal paragraph.\n\n"
        "# ACKNOWLEDGMENTS\n\n"
        "Thank you.\n"
    )
    blocks = parse_normalized_md(md, ARTICLE_ID)
    paras = blocks_of_type(blocks, "paragraph")
    bm = [b for b in blocks if b["block_type"] == "back_matter"]
    assert len(paras) == 1
    assert "Normal paragraph." in paras[0]["text"]
    assert any("Thank you." in b["text"] for b in bm)
