"""Tests for pipeline/step3_build_parents.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.step3_build_parents import (
    PARENT_MAX_TOKENS,
    PARENT_MIN_TOKENS,
    PARENT_TARGET_TOKENS,
    build_parents,
    discover_article_ids,
    process_article,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AID = "pdf_test_parents"
_BIDX = 0


def _reset_bidx():
    global _BIDX
    _BIDX = 0


def _block(
    block_type: str,
    text: str,
    section_path: list,
    block_index: int | None = None,
    heading_level=None,
    char_start: int = 0,
    char_end: int | None = None,
) -> dict:
    global _BIDX
    idx = block_index if block_index is not None else _BIDX
    _BIDX = idx + 1
    return {
        "block_id": f"{AID}_{idx:06d}",
        "article_id": AID,
        "block_index": idx,
        "block_type": block_type,
        "section_path": section_path,
        "heading_level": heading_level,
        "text": text,
        "char_start": char_start,
        "char_end": char_end if char_end is not None else char_start + len(text),
        "token_count": max(1, len(text.split())),
    }


def para(text, section_path, **kw):
    return _block("paragraph", text, section_path, **kw)


def heading(text, section_path, level=1, **kw):
    return _block("heading", text, section_path, heading_level=level, **kw)


def fig_cap(text, section_path, **kw):
    return _block("figure_caption", text, section_path, **kw)


def tbl_cap(text, section_path, **kw):
    return _block("table_caption", text, section_path, **kw)


def ref(text, section_path, **kw):
    return _block("reference", text, section_path, **kw)


def bm(text, section_path, hlevel=None, **kw):
    return _block("back_matter", text, section_path, heading_level=hlevel, **kw)


def parents_of_type(parents, chunk_type):
    return [p for p in parents if p["chunk_type"] == chunk_type]


def _setup_article(tmp_path: Path, article_id: str, blocks: list[dict]) -> None:
    d = tmp_path / "index" / article_id
    d.mkdir(parents=True)
    with (d / "structured_blocks.jsonl").open("w") as f:
        for b in blocks:
            f.write(json.dumps(b) + "\n")


# ---------------------------------------------------------------------------
# Section-path boundary enforcement
# ---------------------------------------------------------------------------

def test_parents_do_not_cross_section_path():
    _reset_bidx()
    blocks = [
        para("Para in intro.", ["Text", "Introduction"]),
        para("Another intro para.", ["Text", "Introduction"]),
        para("Methods paragraph.", ["Text", "Methods"]),
    ]
    parents = build_parents(blocks, AID)
    for p in parents_of_type(parents, "body"):
        # All source blocks must share the same section_path
        paths = set(tuple(b["section_path"]) for b in blocks
                    if b["block_id"] in p["source_block_ids"])
        assert len(paths) == 1, f"Parent {p['parent_id']} crosses section boundaries: {paths}"


def test_different_section_paths_create_separate_parents():
    _reset_bidx()
    blocks = [
        para("Intro text.", ["Text", "Introduction"]),
        para("Methods text.", ["Text", "Methods"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    assert len(body) == 2


def test_same_section_path_merges_paragraphs():
    _reset_bidx()
    blocks = [
        para("First para.", ["Results"]),
        para("Second para.", ["Results"]),
        para("Third para.", ["Results"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    assert len(body) == 1
    assert len(body[0]["source_block_ids"]) == 3


# ---------------------------------------------------------------------------
# Paragraph grouping
# ---------------------------------------------------------------------------

def test_paragraph_blocks_grouped_correctly():
    _reset_bidx()
    blocks = [
        para("A " * 5, ["S1"]),
        para("B " * 5, ["S1"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    assert len(body) == 1
    assert "A A A A A" in body[0]["text"]
    assert "B B B B B" in body[0]["text"]


def test_text_joined_with_double_newline():
    _reset_bidx()
    blocks = [
        para("First.", ["S"]),
        para("Second.", ["S"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    assert body[0]["text"] == "First.\n\nSecond."


def test_single_paragraph_becomes_single_parent():
    _reset_bidx()
    blocks = [para("Only paragraph.", ["S"])]
    parents = build_parents(blocks, AID)
    assert len(parents_of_type(parents, "body")) == 1


# ---------------------------------------------------------------------------
# Source block IDs
# ---------------------------------------------------------------------------

def test_source_block_ids_preserved():
    _reset_bidx()
    blocks = [
        para("A.", ["S"]),
        para("B.", ["S"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")[0]
    expected = {b["block_id"] for b in blocks}
    assert set(body["source_block_ids"]) == expected


def test_source_block_ids_in_order():
    _reset_bidx()
    blocks = [para(f"Para {i}.", ["S"]) for i in range(4)]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")[0]
    ids_in_parent = body["source_block_ids"]
    block_ids_in_order = [b["block_id"] for b in blocks]
    assert ids_in_parent == block_ids_in_order


# ---------------------------------------------------------------------------
# Token limits
# ---------------------------------------------------------------------------

def test_max_tokens_splits_into_two_parents():
    _reset_bidx()
    max_tok = 10
    # Each paragraph is 8 tokens — two won't fit (8+8=16 > 10)
    p1 = para("word " * 8, ["S"])
    p2 = para("word " * 8, ["S"])
    parents = build_parents([p1, p2], AID, parent_max_tokens=max_tok)
    body = parents_of_type(parents, "body")
    assert len(body) == 2


def test_paragraphs_within_limit_merged():
    _reset_bidx()
    max_tok = 20
    blocks = [para("word " * 5, ["S"]), para("word " * 5, ["S"])]  # 5+5=10 ≤ 20
    parents = build_parents(blocks, AID, parent_max_tokens=max_tok)
    body = parents_of_type(parents, "body")
    assert len(body) == 1


def test_exactly_at_limit_merges():
    _reset_bidx()
    max_tok = 10
    blocks = [para("word " * 5, ["S"]), para("word " * 5, ["S"])]  # 5+5=10 == 10
    parents = build_parents(blocks, AID, parent_max_tokens=max_tok)
    body = parents_of_type(parents, "body")
    assert len(body) == 1


def test_one_over_limit_splits():
    _reset_bidx()
    max_tok = 10
    blocks = [para("word " * 5, ["S"]), para("word " * 6, ["S"])]  # 5+6=11 > 10
    parents = build_parents(blocks, AID, parent_max_tokens=max_tok)
    body = parents_of_type(parents, "body")
    assert len(body) == 2


# ---------------------------------------------------------------------------
# Oversized paragraph handling
# ---------------------------------------------------------------------------

def test_oversized_paragraph_flagged():
    _reset_bidx()
    big = para("word " * 2000, ["S"])
    parents = build_parents([big], AID, parent_max_tokens=100)
    body = parents_of_type(parents, "body")
    assert len(body) == 1
    assert body[0]["oversized"] is True


def test_oversized_is_single_block_parent():
    _reset_bidx()
    big = para("word " * 2000, ["S"])
    parents = build_parents([big], AID, parent_max_tokens=100)
    body = parents_of_type(parents, "body")
    assert len(body[0]["source_block_ids"]) == 1


def test_oversized_does_not_merge_with_following():
    _reset_bidx()
    max_tok = 10
    big = para("word " * 100, ["S"])
    small = para("short text.", ["S"])
    parents = build_parents([big, small], AID, parent_max_tokens=max_tok)
    body = parents_of_type(parents, "body")
    assert len(body) == 2
    assert body[0]["oversized"] is True
    assert body[1]["oversized"] is False


def test_normal_blocks_not_flagged_oversized():
    _reset_bidx()
    blocks = [para("small text.", ["S"])]
    parents = build_parents(blocks, AID)
    for p in parents:
        assert p["oversized"] is False


# ---------------------------------------------------------------------------
# Figure / table captions → separate parents
# ---------------------------------------------------------------------------

def test_figure_caption_becomes_separate_parent():
    _reset_bidx()
    blocks = [
        para("Body text.", ["S"]),
        fig_cap("Figure 1. Caption.", ["Captions"]),
    ]
    parents = build_parents(blocks, AID)
    caps = parents_of_type(parents, "figure_caption")
    assert len(caps) == 1


def test_table_caption_becomes_separate_parent():
    _reset_bidx()
    blocks = [
        para("Body text.", ["S"]),
        tbl_cap("Table 1. Summary.", ["Captions"]),
    ]
    parents = build_parents(blocks, AID)
    caps = parents_of_type(parents, "table_caption")
    assert len(caps) == 1


def test_figure_caption_is_single_block():
    _reset_bidx()
    blocks = [fig_cap("Figure 2. A figure.", ["Captions"])]
    parents = build_parents(blocks, AID)
    cap = parents_of_type(parents, "figure_caption")[0]
    assert len(cap["source_block_ids"]) == 1


def test_figure_caption_does_not_merge_into_body():
    _reset_bidx()
    blocks = [
        para("Body.", ["S"]),
        fig_cap("Figure 1. Caption.", ["Captions"]),
        para("More body.", ["S"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    # The two body paragraphs are in different sections — both become parents
    assert all("Figure" not in p["text"] for p in body)


def test_caption_include_in_default_qa_true():
    _reset_bidx()
    blocks = [fig_cap("Figure 1. Caption.", ["Captions"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["include_in_default_qa"] is True


# ---------------------------------------------------------------------------
# References are skipped
# ---------------------------------------------------------------------------

def test_references_not_in_parents():
    _reset_bidx()
    blocks = [
        para("Body paragraph.", ["S"]),
        ref("1. Author et al. (2020).", ["REFERENCES"]),
        ref("2. Author et al. (2021).", ["REFERENCES"]),
    ]
    parents = build_parents(blocks, AID)
    for p in parents:
        assert p["chunk_type"] != "reference"


def test_only_references_produces_no_parents():
    _reset_bidx()
    blocks = [ref("1. Ref one.", ["REFERENCES"]), ref("2. Ref two.", ["REFERENCES"])]
    parents = build_parents(blocks, AID)
    assert len(parents) == 0


def test_references_between_body_paras_do_not_split_parent():
    _reset_bidx()
    # A reference block between two body paras in different sections
    # (refs have their own section path so they don't affect grouping)
    blocks = [
        para("Before.", ["S"]),
        ref("1. Ref.", ["REFERENCES"]),
        para("Also in S.", ["S"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    # Both paras have section_path ["S"], but they are separated by a ref.
    # Since refs are just skipped, the two paras should merge into one parent.
    assert len(body) == 1
    assert len(body[0]["source_block_ids"]) == 2


# ---------------------------------------------------------------------------
# Back matter
# ---------------------------------------------------------------------------

def test_back_matter_content_creates_back_matter_parent():
    _reset_bidx()
    blocks = [bm("We thank the funding agencies.", ["ACKNOWLEDGMENTS"])]
    parents = build_parents(blocks, AID)
    bms = parents_of_type(parents, "back_matter")
    assert len(bms) == 1


def test_back_matter_excluded_from_default_qa():
    _reset_bidx()
    blocks = [bm("Author contributions text.", ["AUTHOR CONTRIBUTIONS"])]
    parents = build_parents(blocks, AID)
    bms = parents_of_type(parents, "back_matter")
    assert all(not p["include_in_default_qa"] for p in bms)


def test_back_matter_heading_skipped():
    _reset_bidx()
    blocks = [
        bm("ACKNOWLEDGMENTS", ["ACKNOWLEDGMENTS"], hlevel=1),
        bm("We thank funding agencies.", ["ACKNOWLEDGMENTS"]),
    ]
    parents = build_parents(blocks, AID)
    bms = parents_of_type(parents, "back_matter")
    # Only the content block should form a parent
    assert len(bms) == 1
    assert bms[0]["text"] == "We thank funding agencies."


def test_back_matter_separate_from_body():
    _reset_bidx()
    blocks = [
        para("Body paragraph.", ["Results"]),
        bm("Acknowledgment text.", ["ACKNOWLEDGMENTS"]),
    ]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    bms = parents_of_type(parents, "back_matter")
    assert len(body) == 1
    assert len(bms) == 1


def test_body_parents_included_in_default_qa():
    _reset_bidx()
    blocks = [para("Important scientific finding.", ["Results"])]
    parents = build_parents(blocks, AID)
    body = parents_of_type(parents, "body")
    assert all(p["include_in_default_qa"] for p in body)


# ---------------------------------------------------------------------------
# Headings are skipped (don't become parents by themselves)
# ---------------------------------------------------------------------------

def test_heading_blocks_skipped():
    _reset_bidx()
    blocks = [
        heading("Methods", ["Methods"]),
        para("Methodology text.", ["Methods"]),
    ]
    parents = build_parents(blocks, AID)
    # No parent should have block_type heading as source
    for p in parents:
        src_ids = set(p["source_block_ids"])
        assert blocks[0]["block_id"] not in src_ids


def test_title_block_skipped():
    _reset_bidx()
    blocks = [
        _block("title", "My Title", ["My Title"], heading_level=1),
        para("Abstract text.", ["My Title"]),
    ]
    parents = build_parents(blocks, AID)
    for p in parents:
        assert blocks[0]["block_id"] not in p["source_block_ids"]


# ---------------------------------------------------------------------------
# char_start / char_end
# ---------------------------------------------------------------------------

def test_char_start_is_first_block_char_start():
    _reset_bidx()
    b1 = para("First.", ["S"], char_start=10, char_end=16)
    b2 = para("Second.", ["S"], char_start=18, char_end=25)
    parents = build_parents([b1, b2], AID)
    body = parents_of_type(parents, "body")[0]
    assert body["char_start"] == 10


def test_char_end_is_last_block_char_end():
    _reset_bidx()
    b1 = para("First.", ["S"], char_start=10, char_end=16)
    b2 = para("Second.", ["S"], char_start=18, char_end=25)
    parents = build_parents([b1, b2], AID)
    body = parents_of_type(parents, "body")[0]
    assert body["char_end"] == 25


def test_single_block_char_span_unchanged():
    _reset_bidx()
    b = para("Only.", ["S"], char_start=50, char_end=55)
    parents = build_parents([b], AID)
    p = parents[0]
    assert p["char_start"] == 50
    assert p["char_end"] == 55


def test_char_start_less_than_char_end():
    _reset_bidx()
    blocks = [para("Some text.", ["S"])]
    parents = build_parents(blocks, AID)
    for p in parents:
        assert p["char_start"] <= p["char_end"]


# ---------------------------------------------------------------------------
# Text not rewritten
# ---------------------------------------------------------------------------

def test_text_not_rewritten():
    _reset_bidx()
    original = "Cancer cells degrade myelin sheaths: p < 0.001 (n=33)."
    blocks = [para(original, ["Results"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["text"] == original


def test_multi_block_text_join_only():
    _reset_bidx()
    t1 = "First sentence."
    t2 = "Second sentence."
    blocks = [para(t1, ["S"]), para(t2, ["S"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["text"] == f"{t1}\n\n{t2}"


def test_special_characters_preserved():
    _reset_bidx()
    original = "β = 0.5; CD8⁺ T cells; p = 3.14 × 10⁻⁶"
    blocks = [para(original, ["Results"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["text"] == original


# ---------------------------------------------------------------------------
# Parent schema completeness
# ---------------------------------------------------------------------------

def test_all_parents_have_required_fields():
    _reset_bidx()
    blocks = [
        para("Body.", ["S"]),
        fig_cap("Figure 1. Cap.", ["Captions"]),
        bm("Acknowledgment.", ["ACKNOWLEDGMENTS"]),
    ]
    parents = build_parents(blocks, AID)
    required = {
        "parent_id", "article_id", "parent_index", "chunk_level", "chunk_type",
        "section_path", "source_block_ids", "text", "char_start", "char_end",
        "token_count", "oversized", "include_in_default_qa",
    }
    for p in parents:
        missing = required - p.keys()
        assert not missing, f"Parent {p['parent_id']} missing: {missing}"


def test_parent_index_sequential():
    _reset_bidx()
    blocks = [para(f"Para {i}.", [f"S{i}"]) for i in range(5)]
    parents = build_parents(blocks, AID)
    for i, p in enumerate(parents):
        assert p["parent_index"] == i


def test_parent_id_format():
    _reset_bidx()
    blocks = [para("Body.", ["S"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["parent_id"] == f"{AID}::parent_000000"


def test_chunk_level_is_parent():
    _reset_bidx()
    blocks = [para("Text.", ["S"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["chunk_level"] == "parent"


def test_token_count_positive():
    _reset_bidx()
    blocks = [para("Some text.", ["S"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["token_count"] >= 1


def test_section_path_taken_from_first_source_block():
    _reset_bidx()
    blocks = [para("Text.", ["Results", "Statistics"])]
    parents = build_parents(blocks, AID)
    assert parents[0]["section_path"] == ["Results", "Statistics"]


# ---------------------------------------------------------------------------
# process_article — integration
# ---------------------------------------------------------------------------

def test_process_article_creates_parents_jsonl(tmp_path):
    _reset_bidx()
    blocks = [
        para("Para one.", ["S"]),
        para("Para two.", ["S"]),
        fig_cap("Figure 1. Cap.", ["Captions"]),
    ]
    _setup_article(tmp_path, AID, blocks)
    result = process_article(AID, tmp_path)
    assert not result.get("skipped")
    out = tmp_path / "index" / AID / "parents.jsonl"
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == result["parent_count"]


def test_process_article_creates_manifest(tmp_path):
    _reset_bidx()
    _setup_article(tmp_path, AID, [para("Text.", ["S"])])
    process_article(AID, tmp_path)
    m = json.loads((tmp_path / "index" / AID / "parent_manifest.json").read_text())
    assert m["article_id"] == AID
    assert "chunk_type_counts" in m
    assert "default_parameters" in m
    assert "oversized_parent_count" in m


def test_process_article_skips_existing(tmp_path):
    _reset_bidx()
    _setup_article(tmp_path, AID, [para("Text.", ["S"])])
    process_article(AID, tmp_path)
    result = process_article(AID, tmp_path, overwrite=False)
    assert result.get("skipped")


def test_process_article_overwrite(tmp_path):
    _reset_bidx()
    _setup_article(tmp_path, AID, [para("Text.", ["S"])])
    process_article(AID, tmp_path)
    result = process_article(AID, tmp_path, overwrite=True)
    assert not result.get("skipped")


def test_process_article_missing_blocks(tmp_path):
    (tmp_path / "index" / AID).mkdir(parents=True)
    result = process_article(AID, tmp_path)
    assert result.get("skipped")


def test_jsonl_lines_valid_json(tmp_path):
    _reset_bidx()
    _setup_article(tmp_path, AID, [para("Text.", ["S"]), para("More.", ["S"])])
    process_article(AID, tmp_path)
    out = tmp_path / "index" / AID / "parents.jsonl"
    for line in out.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            assert isinstance(obj, dict)


# ---------------------------------------------------------------------------
# discover_article_ids
# ---------------------------------------------------------------------------

def test_discover_finds_indexed_articles(tmp_path):
    for aid in ("pdf_aaa", "pdf_bbb"):
        d = tmp_path / "index" / aid
        d.mkdir(parents=True)
        (d / "structured_blocks.jsonl").write_text("")
    ids = discover_article_ids(tmp_path)
    assert "pdf_aaa" in ids and "pdf_bbb" in ids


def test_discover_returns_sorted(tmp_path):
    for aid in ("pdf_zzz", "pdf_aaa", "pdf_mmm"):
        d = tmp_path / "index" / aid
        d.mkdir(parents=True)
        (d / "structured_blocks.jsonl").write_text("")
    ids = discover_article_ids(tmp_path)
    assert ids == sorted(ids)


def test_discover_empty(tmp_path):
    (tmp_path / "index").mkdir()
    assert discover_article_ids(tmp_path) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_blocks_list():
    parents = build_parents([], AID)
    assert parents == []


def test_only_headings_produces_no_parents():
    _reset_bidx()
    blocks = [heading("Methods", ["Methods"]), heading("Results", ["Results"])]
    parents = build_parents(blocks, AID)
    assert len(parents) == 0


def test_mixed_types_correct_counts():
    _reset_bidx()
    blocks = [
        para("Body.", ["S"]),
        ref("1. Ref.", ["REFS"]),
        fig_cap("Figure 1. Cap.", ["Captions"]),
        bm("Ack text.", ["ACKNOWLEDGMENTS"]),
        tbl_cap("Table 1. T.", ["Captions"]),
    ]
    parents = build_parents(blocks, AID)
    assert len(parents_of_type(parents, "body")) == 1
    assert len(parents_of_type(parents, "figure_caption")) == 1
    assert len(parents_of_type(parents, "table_caption")) == 1
    assert len(parents_of_type(parents, "back_matter")) == 1
    # references → 0
    assert len(parents_of_type(parents, "reference")) == 0
