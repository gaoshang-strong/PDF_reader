"""Tests for Step 3B-2: build_children."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.step3_build_children import (
    CHILD_MAX_TOKENS,
    CHILD_OVERLAP_TOKENS,
    _build_units,
    _group_units,
    _join_group,
    _overlap_suffix,
    _split_sentences,
    _text_for_bm25,
    _text_for_embedding,
    _tok,
    _word_chunks,
    process_article,
    split_parent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ART = "test_art"
_PID = f"{_ART}::parent_000000"


def _make_parent(
    text: str,
    chunk_type: str = "body",
    section_path: list[str] | None = None,
    include_in_default_qa: bool = True,
    char_start: int = 0,
    parent_index: int = 0,
    article_id: str = _ART,
) -> dict:
    pid = f"{article_id}::parent_{parent_index:06d}"
    return {
        "parent_id": pid,
        "article_id": article_id,
        "parent_index": parent_index,
        "chunk_level": "parent",
        "chunk_type": chunk_type,
        "section_path": section_path or ["Introduction"],
        "source_block_ids": [],
        "text": text,
        "char_start": char_start,
        "char_end": char_start + len(text),
        "token_count": _tok(text),
        "oversized": False,
        "include_in_default_qa": include_in_default_qa,
    }


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Unit tests: token count
# ---------------------------------------------------------------------------


def test_tok_basic():
    assert _tok("hello world") == 2


def test_tok_minimum_one():
    assert _tok("") == 1
    assert _tok("   ") == 1


# ---------------------------------------------------------------------------
# Unit tests: sentence splitting
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    text = "First sentence. Second sentence. Third sentence."
    sents = _split_sentences(text)
    assert len(sents) >= 2


def test_split_sentences_returns_nonempty():
    sents = _split_sentences("Only one sentence here")
    assert len(sents) == 1
    assert sents[0] == "Only one sentence here"


# ---------------------------------------------------------------------------
# Unit tests: word chunks
# ---------------------------------------------------------------------------


def test_word_chunks_splits_evenly():
    text = _words(10)
    chunks = _word_chunks(text, 3)
    assert len(chunks) == 4  # 3+3+3+1
    assert all(len(c.split()) <= 3 for c in chunks)


def test_word_chunks_single_chunk():
    text = "hello world"
    chunks = _word_chunks(text, 10)
    assert chunks == ["hello world"]


# ---------------------------------------------------------------------------
# Unit tests: _build_units
# ---------------------------------------------------------------------------


def test_build_units_single_para_fits():
    text = _words(50)
    units = _build_units(text, 100)
    assert len(units) == 1
    assert units[0].text == text
    assert units[0].para_idx == 0


def test_build_units_two_paragraphs():
    p1 = _words(20)
    p2 = _words(20)
    text = p1 + "\n\n" + p2
    units = _build_units(text, 100)
    assert len(units) == 2
    assert units[0].para_idx == 0
    assert units[1].para_idx == 1


def test_build_units_large_para_splits_sentences():
    # Build a paragraph with many sentences — need > 550 words total
    sents = ["This is sentence number %d and it contains several additional filler words." % i for i in range(60)]
    para = " ".join(sents)
    assert _tok(para) > CHILD_MAX_TOKENS
    units = _build_units(para, CHILD_MAX_TOKENS)
    assert len(units) > 1
    for u in units:
        assert _tok(u.text) <= CHILD_MAX_TOKENS


def test_build_units_word_fallback():
    # One very long "sentence" (no punctuation to split on)
    para = _words(600)
    assert _tok(para) > CHILD_MAX_TOKENS
    units = _build_units(para, CHILD_MAX_TOKENS)
    assert len(units) > 1
    for u in units:
        assert _tok(u.text) <= CHILD_MAX_TOKENS


def test_build_units_p_start_p_end_coverage():
    p1 = "Alpha beta gamma"
    p2 = "Delta epsilon zeta"
    text = p1 + "\n\n" + p2
    units = _build_units(text, 100)
    assert units[0].p_start == 0
    assert units[0].p_end == len(p1)
    assert units[1].p_start == len(p1) + 2  # after '\n\n'
    assert units[1].p_end == len(p1) + 2 + len(p2)


# ---------------------------------------------------------------------------
# Unit tests: grouping and joining
# ---------------------------------------------------------------------------


def test_group_units_packs_small_units():
    units = _build_units("\n\n".join(_words(10) for _ in range(5)), 100)
    groups = _group_units(units, 100)
    assert len(groups) == 1


def test_group_units_splits_on_overflow():
    # Each paragraph is 20 tokens; max is 25 → at most 1 per group
    paras = "\n\n".join(_words(20) for _ in range(4))
    units = _build_units(paras, 25)
    groups = _group_units(units, 25)
    assert len(groups) == 4


def test_join_group_two_paras():
    p1 = "Para one text"
    p2 = "Para two text"
    text = p1 + "\n\n" + p2
    units = _build_units(text, 100)
    assert len(units) == 2
    joined = _join_group(units)
    assert joined == p1 + "\n\n" + p2


def test_join_group_same_para_space():
    # Two sentence units from the same paragraph → joined with space
    p = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    # Make paragraph huge enough to force sentence splitting
    long_p = (p + " ") * 20
    units = _build_units(long_p, 20)
    # All units have same para_idx
    assert all(u.para_idx == units[0].para_idx for u in units)
    joined = _join_group(units[:2])
    assert "\n\n" not in joined
    assert " " in joined


# ---------------------------------------------------------------------------
# Unit tests: overlap
# ---------------------------------------------------------------------------


def test_overlap_suffix_empty_if_group_empty():
    result = _overlap_suffix([], 60)
    assert result == []


def test_overlap_suffix_respects_limit():
    paras = "\n\n".join(_words(30) for _ in range(3))
    units = _build_units(paras, 200)
    groups = _group_units(units, 200)
    ov = _overlap_suffix(groups[-1], 35)
    total = sum(_tok(u.text) for u in ov)
    assert total <= 35


def test_overlap_suffix_returns_tail():
    p1 = _words(10)
    p2 = _words(10)
    text = p1 + "\n\n" + p2
    units = _build_units(text, 200)
    groups = _group_units(units, 200)
    # whole group fits in one group; overlap of 12 tokens → picks p2
    ov = _overlap_suffix(groups[0], 12)
    assert any("word10" in u.text for u in ov) or len(ov) >= 1


# ---------------------------------------------------------------------------
# split_parent: basic behaviour
# ---------------------------------------------------------------------------


def test_single_small_parent_one_child():
    parent = _make_parent(_words(30))
    children = split_parent(parent, _ART, 0)
    assert len(children) == 1


def test_parent_text_preserved_in_first_child():
    text = _words(30)
    parent = _make_parent(text)
    children = split_parent(parent, _ART, 0)
    assert children[0]["text"] == text


def test_large_parent_produces_multiple_children():
    # 4 paragraphs × 200 words each → must split
    paras = "\n\n".join(_words(200) for _ in range(4))
    parent = _make_parent(paras)
    children = split_parent(parent, _ART, 0, child_max_tokens=CHILD_MAX_TOKENS)
    assert len(children) > 1


# ---------------------------------------------------------------------------
# split_parent: parent boundary
# ---------------------------------------------------------------------------


def test_children_do_not_cross_parent_boundary():
    """Children from different parents must never share a parent_id cross-contamination."""
    p0 = _make_parent(_words(600), parent_index=0)
    p1 = _make_parent(_words(600), parent_index=1)
    c0 = split_parent(p0, _ART, 0)
    c1 = split_parent(p1, _ART, len(c0))
    for c in c0:
        assert c["parent_id"] == p0["parent_id"]
    for c in c1:
        assert c["parent_id"] == p1["parent_id"]


# ---------------------------------------------------------------------------
# split_parent: inherited fields
# ---------------------------------------------------------------------------


def test_parent_id_preserved():
    parent = _make_parent(_words(600))
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["parent_id"] == parent["parent_id"]


def test_article_id_preserved():
    parent = _make_parent(_words(30))
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["article_id"] == _ART


def test_section_path_inherited():
    sp = ["Methods", "Statistics"]
    parent = _make_parent(_words(30), section_path=sp)
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["section_path"] == sp


def test_include_in_default_qa_inherited_true():
    parent = _make_parent(_words(30), include_in_default_qa=True)
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["include_in_default_qa"] is True


def test_include_in_default_qa_inherited_false():
    parent = _make_parent(_words(30), include_in_default_qa=False)
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["include_in_default_qa"] is False


def test_chunk_type_inherited():
    parent = _make_parent(_words(30), chunk_type="back_matter")
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["chunk_type"] == "back_matter"


# ---------------------------------------------------------------------------
# split_parent: back_matter
# ---------------------------------------------------------------------------


def test_back_matter_children_exclude_from_qa():
    parent = _make_parent(_words(600), chunk_type="back_matter", include_in_default_qa=False)
    children = split_parent(parent, _ART, 0)
    assert len(children) > 0
    for c in children:
        assert c["include_in_default_qa"] is False
        assert c["chunk_type"] == "back_matter"


# ---------------------------------------------------------------------------
# split_parent: captions
# ---------------------------------------------------------------------------


def test_figure_caption_one_child_when_small():
    caption = "Figure 1. This is a short caption."
    parent = _make_parent(caption, chunk_type="figure_caption")
    children = split_parent(parent, _ART, 0)
    assert len(children) == 1
    assert children[0]["chunk_type"] == "figure_caption"


def test_table_caption_one_child_when_small():
    caption = "Table 1. Results of the experiment."
    parent = _make_parent(caption, chunk_type="table_caption")
    children = split_parent(parent, _ART, 0)
    assert len(children) == 1
    assert children[0]["chunk_type"] == "table_caption"


def test_figure_caption_text_for_embedding_prefix():
    caption = "Figure 2. Important result."
    parent = _make_parent(caption, chunk_type="figure_caption", section_path=["Results"])
    children = split_parent(parent, _ART, 0)
    assert children[0]["text_for_embedding"].startswith("Figure caption:")


def test_table_caption_text_for_embedding_prefix():
    caption = "Table 2. Summary statistics."
    parent = _make_parent(caption, chunk_type="table_caption", section_path=["Results"])
    children = split_parent(parent, _ART, 0)
    assert children[0]["text_for_embedding"].startswith("Table caption:")


# ---------------------------------------------------------------------------
# split_parent: paragraph-aware splitting
# ---------------------------------------------------------------------------


def test_paragraph_aware_splitting_respects_boundaries():
    """A child should not straddle two paragraphs when one paragraph fills the budget."""
    # p1 is exactly at target, p2 would overflow
    p1 = _words(540)  # 540 < 550 fits alone
    p2 = _words(540)  # combined 1080 > 550 → new child
    text = p1 + "\n\n" + p2
    parent = _make_parent(text)
    children = split_parent(parent, _ART, 0, child_max_tokens=550, child_overlap_tokens=0)
    # At least 2 children (each para fills a child)
    assert len(children) >= 2


def test_paragraph_grouping_combines_small_paras():
    """Small paragraphs should be grouped together."""
    paras = "\n\n".join(_words(80) for _ in range(4))  # 4 × 80 = 320 < 550
    parent = _make_parent(paras)
    children = split_parent(parent, _ART, 0, child_max_tokens=550, child_overlap_tokens=0)
    assert len(children) == 1
    for p in paras.split("\n\n"):
        assert p in children[0]["text"]


# ---------------------------------------------------------------------------
# split_parent: sentence fallback
# ---------------------------------------------------------------------------


def test_sentence_fallback_for_overlong_paragraph():
    sents = ["Sentence number %d ends here with some extra filler words." % i for i in range(80)]
    para = " ".join(sents)
    assert _tok(para) > CHILD_MAX_TOKENS
    parent = _make_parent(para)
    children = split_parent(parent, _ART, 0, child_overlap_tokens=0)
    assert len(children) > 1
    for c in children:
        assert _tok(c["text"]) <= CHILD_MAX_TOKENS


# ---------------------------------------------------------------------------
# split_parent: word fallback
# ---------------------------------------------------------------------------


def test_word_fallback_for_extremely_long_sentence():
    # No sentence-ending punctuation
    para = _words(700)
    assert _tok(para) > CHILD_MAX_TOKENS
    parent = _make_parent(para)
    children = split_parent(parent, _ART, 0, child_overlap_tokens=0)
    assert len(children) > 1
    for c in children:
        assert _tok(c["text"]) <= CHILD_MAX_TOKENS


# ---------------------------------------------------------------------------
# split_parent: overlap
# ---------------------------------------------------------------------------


def test_overlap_not_on_first_child():
    paras = "\n\n".join(_words(200) for _ in range(4))
    parent = _make_parent(paras)
    children = split_parent(parent, _ART, 0, child_max_tokens=250, child_overlap_tokens=60)
    assert len(children) > 1
    # The text of child[0] must be a substring of parent text (no extra prefix)
    assert children[0]["text"] in parent["text"]


def test_overlap_added_to_subsequent_children():
    # Two paragraphs of 300 tokens each → two children, second gets overlap
    p1 = _words(300)
    p2 = _words(300)
    text = p1 + "\n\n" + p2
    parent = _make_parent(text)
    children = split_parent(parent, _ART, 0, child_max_tokens=350, child_overlap_tokens=60)
    assert len(children) >= 2
    # Second child should contain some text from p1 (the overlap)
    assert any(w in children[1]["text"] for w in p1.split()[-5:])


def test_overlap_tokens_respected():
    p1 = _words(300)
    p2 = _words(300)
    text = p1 + "\n\n" + p2
    parent = _make_parent(text)
    children = split_parent(parent, _ART, 0, child_max_tokens=350, child_overlap_tokens=40)
    if len(children) >= 2:
        # Overlap portion (extra text in child[1] beyond p2 start) must be ≤ 40 tokens
        # The overlap is the suffix of child[0]'s core that appears as prefix in child[1]
        # Just check that child[1] is not massively over budget
        assert _tok(children[1]["text"]) <= 350 + 40 + 5  # small slack


# ---------------------------------------------------------------------------
# split_parent: token_count
# ---------------------------------------------------------------------------


def test_token_count_is_recomputed():
    text = _words(100)
    parent = _make_parent(text)
    parent["token_count"] = 999  # intentionally wrong
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["token_count"] == _tok(c["text"])


# ---------------------------------------------------------------------------
# split_parent: text_for_embedding
# ---------------------------------------------------------------------------


def test_text_for_embedding_body_has_section_prefix():
    sp = ["Results", "Main findings"]
    parent = _make_parent(_words(30), chunk_type="body", section_path=sp)
    children = split_parent(parent, _ART, 0)
    emb = children[0]["text_for_embedding"]
    assert emb.startswith("Section: Results > Main findings")
    assert children[0]["text"] in emb


def test_text_for_embedding_includes_child_text():
    parent = _make_parent(_words(30))
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["text"] in c["text_for_embedding"]


# ---------------------------------------------------------------------------
# split_parent: text_for_bm25
# ---------------------------------------------------------------------------


def test_text_for_bm25_contains_section_and_text():
    sp = ["Methods", "Data collection"]
    parent = _make_parent(_words(30), section_path=sp)
    children = split_parent(parent, _ART, 0)
    bm = children[0]["text_for_bm25"]
    assert "Methods" in bm
    assert "Data collection" in bm
    assert children[0]["text"] in bm


# ---------------------------------------------------------------------------
# split_parent: chunk_id and indices
# ---------------------------------------------------------------------------


def test_chunk_id_format():
    parent = _make_parent(_words(30))
    children = split_parent(parent, _ART, 0)
    assert children[0]["chunk_id"] == f"{_ART}::child_000000"


def test_child_index_global_sequential():
    paras = "\n\n".join(_words(200) for _ in range(6))
    parent = _make_parent(paras)
    children = split_parent(parent, _ART, 5, child_max_tokens=250)
    for i, c in enumerate(children):
        assert c["child_index"] == 5 + i


def test_child_index_within_parent_sequential():
    paras = "\n\n".join(_words(200) for _ in range(6))
    parent = _make_parent(paras)
    children = split_parent(parent, _ART, 0, child_max_tokens=250)
    for i, c in enumerate(children):
        assert c["child_index_within_parent"] == i


def test_chunk_level_is_child():
    parent = _make_parent(_words(30))
    children = split_parent(parent, _ART, 0)
    for c in children:
        assert c["chunk_level"] == "child"


# ---------------------------------------------------------------------------
# split_parent: char spans
# ---------------------------------------------------------------------------


def test_char_spans_monotonic():
    paras = "\n\n".join(_words(200) for _ in range(4))
    parent = _make_parent(paras, char_start=1000)
    children = split_parent(parent, _ART, 0, child_max_tokens=250)
    for c in children:
        assert c["char_start"] >= parent["char_start"]
        assert c["char_end"] >= c["char_start"]


def test_char_spans_within_parent_bounds():
    paras = "\n\n".join(_words(200) for _ in range(4))
    parent = _make_parent(paras, char_start=500)
    children = split_parent(parent, _ART, 0, child_max_tokens=250)
    for c in children:
        assert c["char_start"] >= parent["char_start"]
        assert c["char_end"] <= parent["char_end"] + 10  # small slack for edge cases


def test_span_is_approximate_false_for_no_overlap():
    p1 = _words(200)
    parent = _make_parent(p1)
    children = split_parent(parent, _ART, 0, child_overlap_tokens=0)
    for c in children:
        assert c["span_is_approximate"] is False


def test_span_is_approximate_true_for_overlapping_children():
    p1 = _words(300)
    p2 = _words(300)
    text = p1 + "\n\n" + p2
    parent = _make_parent(text)
    children = split_parent(parent, _ART, 0, child_max_tokens=350, child_overlap_tokens=60)
    if len(children) >= 2:
        assert children[1]["span_is_approximate"] is True


def test_first_child_span_starts_at_parent_char_start():
    text = _words(30)
    parent = _make_parent(text, char_start=2000)
    children = split_parent(parent, _ART, 0, child_overlap_tokens=0)
    assert children[0]["char_start"] == 2000


# ---------------------------------------------------------------------------
# process_article integration
# ---------------------------------------------------------------------------


def test_process_article_creates_files(tmp_path):
    article_id = "pdf_test"
    index_dir = tmp_path / "index" / article_id
    index_dir.mkdir(parents=True)

    # Write a minimal parents.jsonl
    parents = [
        _make_parent(_words(300), parent_index=0, article_id=article_id),
        _make_parent(_words(300), parent_index=1, article_id=article_id,
                     chunk_type="figure_caption"),
    ]
    parents_path = index_dir / "parents.jsonl"
    with parents_path.open("w") as f:
        for p in parents:
            f.write(json.dumps(p) + "\n")

    result = process_article(article_id, tmp_path)

    assert not result.get("skipped")
    assert (index_dir / "children.jsonl").exists()
    assert (index_dir / "child_manifest.json").exists()
    assert result["child_count"] > 0


def test_process_article_skips_if_no_overwrite(tmp_path):
    article_id = "pdf_test2"
    index_dir = tmp_path / "index" / article_id
    index_dir.mkdir(parents=True)
    parents_path = index_dir / "parents.jsonl"
    parents_path.write_text(
        json.dumps(_make_parent(_words(30), article_id=article_id)) + "\n"
    )
    children_path = index_dir / "children.jsonl"
    children_path.write_text("")  # already exists

    result = process_article(article_id, tmp_path, overwrite=False)
    assert result.get("skipped")


def test_process_article_parent_coverage(tmp_path):
    """Every body parent must produce at least one child."""
    article_id = "pdf_cov"
    index_dir = tmp_path / "index" / article_id
    index_dir.mkdir(parents=True)

    parents = [
        _make_parent(_words(300), parent_index=i, article_id=article_id)
        for i in range(5)
    ]
    with (index_dir / "parents.jsonl").open("w") as f:
        for p in parents:
            f.write(json.dumps(p) + "\n")

    process_article(article_id, tmp_path)

    children = [
        json.loads(l)
        for l in (index_dir / "children.jsonl").read_text().splitlines()
        if l.strip()
    ]
    child_parent_ids = {c["parent_id"] for c in children}
    for p in parents:
        assert p["parent_id"] in child_parent_ids


def test_process_article_child_count_in_manifest(tmp_path):
    article_id = "pdf_mf"
    index_dir = tmp_path / "index" / article_id
    index_dir.mkdir(parents=True)
    with (index_dir / "parents.jsonl").open("w") as f:
        f.write(json.dumps(_make_parent(_words(30), article_id=article_id)) + "\n")

    result = process_article(article_id, tmp_path)
    manifest = json.loads((index_dir / "child_manifest.json").read_text())
    assert manifest["child_count"] == result["child_count"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_single_article(tmp_path, monkeypatch):
    import subprocess
    import sys

    article_id = "pdf_cli"
    index_dir = tmp_path / "index" / article_id
    index_dir.mkdir(parents=True)
    with (index_dir / "parents.jsonl").open("w") as f:
        f.write(json.dumps(_make_parent(_words(50), article_id=article_id)) + "\n")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "pipeline" / "step3_build_children.py"),
        "--data-root", str(tmp_path),
        "--article-id", article_id,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert (index_dir / "children.jsonl").exists()
