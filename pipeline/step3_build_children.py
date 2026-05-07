#!/usr/bin/env python3
"""Step 3B-2: Build child chunks from Step 3B-1 parent chunks.

No LLM or external API. Text is never rewritten.
This step only splits existing parent text into smaller retrieval chunks.

Reads:  data/index/{article_id}/parents.jsonl
Writes: data/index/{article_id}/children.jsonl
        data/index/{article_id}/child_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CHILD_TARGET_TOKENS = 380
CHILD_MAX_TOKENS = 550
CHILD_MIN_TOKENS = 120
CHILD_OVERLAP_TOKENS = 60

BUILDER_VERSION = "3b2.1"

# Sentence boundary: end-punctuation followed by whitespace then uppercase / open-quote
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\(])')


# ---------------------------------------------------------------------------
# Token count
# ---------------------------------------------------------------------------

def _tok(text: str) -> int:
    return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# Sentence / word splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _word_chunks(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + max_tokens]))
        i += max_tokens
    return chunks or [text]


# ---------------------------------------------------------------------------
# Unit dataclass
# ---------------------------------------------------------------------------

class _Unit(NamedTuple):
    text: str
    para_idx: int   # which paragraph this unit came from
    p_start: int    # byte offset of the raw paragraph in parent_text
    p_end: int


def _build_units(parent_text: str, child_max_tokens: int) -> list[_Unit]:
    """Decompose parent text into atomic units (paragraph → sentence → word level)."""
    units: list[_Unit] = []
    pos = 0
    para_idx = 0

    for raw_para in parent_text.split("\n\n"):
        p_start = pos
        p_end = pos + len(raw_para)
        pos = p_end + 2  # skip the '\n\n' separator

        para = raw_para.strip()
        if not para:
            para_idx += 1
            continue

        if _tok(para) <= child_max_tokens:
            units.append(_Unit(para, para_idx, p_start, p_end))
        else:
            sents = _split_sentences(para)
            if len(sents) <= 1:
                # Single oversized sentence — word split
                for chunk in _word_chunks(para, child_max_tokens):
                    units.append(_Unit(chunk, para_idx, p_start, p_end))
            else:
                for sent in sents:
                    if _tok(sent) <= child_max_tokens:
                        units.append(_Unit(sent, para_idx, p_start, p_end))
                    else:
                        for chunk in _word_chunks(sent, child_max_tokens):
                            units.append(_Unit(chunk, para_idx, p_start, p_end))

        para_idx += 1

    return units


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _group_units(units: list[_Unit], child_max_tokens: int) -> list[list[_Unit]]:
    """Pack units into groups not exceeding child_max_tokens."""
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tok = 0

    for u in units:
        tok = _tok(u.text)
        if current and current_tok + tok > child_max_tokens:
            groups.append(current)
            current = []
            current_tok = 0
        current.append(u)
        current_tok += tok

    if current:
        groups.append(current)

    return groups


def _join_group(units: list[_Unit]) -> str:
    """Join units back to text, using \\n\\n between paragraphs, space within a paragraph."""
    if not units:
        return ""
    parts = [units[0].text]
    for i in range(1, len(units)):
        sep = "\n\n" if units[i].para_idx != units[i - 1].para_idx else " "
        parts.append(sep + units[i].text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def _overlap_suffix(group: list[_Unit], overlap_tokens: int) -> list[_Unit]:
    """Return the tail of *group* fitting within *overlap_tokens*.

    Prefers whole units (paragraph or sentence).  If the last unit alone
    exceeds the budget, falls back to a word-level suffix of that unit.
    Returns an empty list when *overlap_tokens* is zero or group is empty.
    """
    if overlap_tokens <= 0 or not group:
        return []
    selected: list[_Unit] = []
    total = 0
    for u in reversed(group):
        tok = _tok(u.text)
        if total + tok > overlap_tokens:
            if not selected:
                # Last unit alone is too big — word-level fallback
                words = u.text.split()
                suffix = " ".join(words[-overlap_tokens:])
                selected = [_Unit(suffix, u.para_idx, u.p_start, u.p_end)]
            break
        selected.insert(0, u)
        total += tok
    return selected


# ---------------------------------------------------------------------------
# Text field builders
# ---------------------------------------------------------------------------

def _section_prefix(section_path: list[str]) -> str:
    return " > ".join(section_path)


def _text_for_embedding(text: str, section_path: list[str], chunk_type: str) -> str:
    prefix = _section_prefix(section_path)
    if chunk_type == "figure_caption":
        return f"Figure caption: {prefix}\n\n{text}"
    if chunk_type == "table_caption":
        return f"Table caption: {prefix}\n\n{text}"
    return f"Section: {prefix}\n\n{text}"


def _text_for_bm25(text: str, section_path: list[str]) -> str:
    return " ".join(section_path) + " " + text


# ---------------------------------------------------------------------------
# Core split function
# ---------------------------------------------------------------------------

def split_parent(
    parent: dict,
    article_id: str,
    start_child_index: int,
    child_max_tokens: int = CHILD_MAX_TOKENS,
    child_overlap_tokens: int = CHILD_OVERLAP_TOKENS,
) -> list[dict]:
    """Split one parent into children with overlap."""
    parent_text: str = parent["text"]
    units = _build_units(parent_text, child_max_tokens)
    if not units:
        return []

    groups = _group_units(units, child_max_tokens)
    children: list[dict] = []
    child_index = start_child_index
    scan_pos = 0  # forward pointer into parent_text for substring search

    for seg_i, core_units in enumerate(groups):
        # --- build overlap prefix ---
        overlap_units = (
            _overlap_suffix(groups[seg_i - 1], child_overlap_tokens) if seg_i > 0 else []
        )

        core_text = _join_group(core_units)
        if overlap_units:
            ov_text = _join_group(overlap_units)
            sep = "\n\n" if overlap_units[-1].para_idx != core_units[0].para_idx else " "
            child_text = ov_text + sep + core_text
            search_start = max(0, scan_pos - len(ov_text) - 5)
        else:
            child_text = core_text
            search_start = scan_pos

        # --- character span via deterministic substring search ---
        offset = parent_text.find(child_text, search_start)
        if offset < 0:
            # Try from beginning of remaining text (shouldn't usually happen)
            offset = parent_text.find(child_text, 0)

        if offset >= 0 and not overlap_units:
            char_start = parent["char_start"] + offset
            char_end = parent["char_start"] + offset + len(child_text)
            span_is_approximate = False
        elif offset >= 0 and overlap_units:
            # Substring found but overlap means the span deliberately covers prior text
            char_start = parent["char_start"] + offset
            char_end = parent["char_start"] + offset + len(child_text)
            span_is_approximate = True
        else:
            # Fallback: use paragraph-level bounds and mark approximate
            ov_p_start = overlap_units[0].p_start if overlap_units else core_units[0].p_start
            char_start = parent["char_start"] + ov_p_start
            char_end = parent["char_start"] + core_units[-1].p_end
            span_is_approximate = True

        # Advance scan_pos past the core text (not the overlap)
        core_offset = parent_text.find(core_text, scan_pos)
        if core_offset >= 0:
            scan_pos = core_offset + len(core_text)
        else:
            scan_pos = min(scan_pos + len(core_text), len(parent_text))

        # --- assemble child ---
        section_path: list[str] = parent["section_path"]
        chunk_type: str = parent["chunk_type"]

        children.append(
            {
                "chunk_id": f"{article_id}::child_{child_index:06d}",
                "parent_id": parent["parent_id"],
                "article_id": article_id,
                "child_index": child_index,
                "child_index_within_parent": seg_i,
                "chunk_level": "child",
                "chunk_type": chunk_type,
                "section_path": section_path,
                "text": child_text,
                "text_for_embedding": _text_for_embedding(child_text, section_path, chunk_type),
                "text_for_bm25": _text_for_bm25(child_text, section_path),
                "char_start": char_start,
                "char_end": char_end,
                "token_count": _tok(child_text),
                "oversized": _tok(child_text) > child_max_tokens,
                "include_in_default_qa": parent["include_in_default_qa"],
                "span_is_approximate": span_is_approximate,
            }
        )
        child_index += 1

    return children


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(
    article_id: str,
    children: list[dict],
    input_path: Path,
    output_path: Path,
    params: dict,
) -> dict:
    type_counts = Counter(c["chunk_type"] for c in children)
    oversized = sum(1 for c in children if c.get("oversized"))
    approx = sum(1 for c in children if c.get("span_is_approximate"))
    return {
        "article_id": article_id,
        "builder_version": BUILDER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "child_count": len(children),
        "chunk_type_counts": dict(type_counts),
        "default_parameters": params,
        "oversized_child_count": oversized,
        "approximate_span_count": approx,
    }


# ---------------------------------------------------------------------------
# Article processing
# ---------------------------------------------------------------------------

def process_article(
    article_id: str,
    data_root: Path,
    overwrite: bool = False,
    child_target_tokens: int = CHILD_TARGET_TOKENS,
    child_max_tokens: int = CHILD_MAX_TOKENS,
    child_min_tokens: int = CHILD_MIN_TOKENS,
    child_overlap_tokens: int = CHILD_OVERLAP_TOKENS,
) -> dict:
    index_dir = data_root / "index" / article_id
    parents_path = index_dir / "parents.jsonl"
    children_path = index_dir / "children.jsonl"
    manifest_path = index_dir / "child_manifest.json"

    if not parents_path.exists():
        return {
            "article_id": article_id,
            "skipped": True,
            "reason": f"parents.jsonl not found: {parents_path}",
        }

    if children_path.exists() and not overwrite:
        return {"article_id": article_id, "skipped": True, "reason": "output exists"}

    parents = [
        json.loads(line)
        for line in parents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parents.sort(key=lambda p: p["parent_index"])

    children: list[dict] = []
    for parent in parents:
        children.extend(
            split_parent(
                parent,
                article_id,
                len(children),
                child_max_tokens=child_max_tokens,
                child_overlap_tokens=child_overlap_tokens,
            )
        )

    with children_path.open("w", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    params = {
        "child_target_tokens": child_target_tokens,
        "child_max_tokens": child_max_tokens,
        "child_min_tokens": child_min_tokens,
        "child_overlap_tokens": child_overlap_tokens,
    }
    manifest = _build_manifest(article_id, children, parents_path, children_path, params)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "article_id": article_id,
        "skipped": False,
        "child_count": len(children),
        "chunk_type_counts": manifest["chunk_type_counts"],
        "oversized": manifest["oversized_child_count"],
        "approximate_spans": manifest["approximate_span_count"],
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_article_ids(data_root: Path) -> list[str]:
    index_dir = data_root / "index"
    if not index_dir.exists():
        return []
    return sorted(
        d.name
        for d in index_dir.iterdir()
        if d.is_dir() and (d / "parents.jsonl").exists()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3B-2: Build child chunks from parent chunks (no LLM)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--article-id", help="Process a single article ID")
    parser.add_argument("--all", action="store_true", help="Process all indexed articles")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--child-max-tokens", type=int, default=CHILD_MAX_TOKENS)
    parser.add_argument("--child-target-tokens", type=int, default=CHILD_TARGET_TOKENS)
    parser.add_argument("--child-min-tokens", type=int, default=CHILD_MIN_TOKENS)
    parser.add_argument("--child-overlap-tokens", type=int, default=CHILD_OVERLAP_TOKENS)
    args = parser.parse_args()

    if args.article_id:
        article_ids = [args.article_id]
    else:
        article_ids = discover_article_ids(args.data_root)

    if not article_ids:
        print("No indexed articles found.")
        return

    print(f"Building children for {len(article_ids)} article(s) ...")

    for article_id in article_ids:
        try:
            result = process_article(
                article_id,
                args.data_root,
                overwrite=args.overwrite,
                child_target_tokens=args.child_target_tokens,
                child_max_tokens=args.child_max_tokens,
                child_min_tokens=args.child_min_tokens,
                child_overlap_tokens=args.child_overlap_tokens,
            )
            if result.get("skipped"):
                print(f"  {article_id}: SKIPPED  ({result['reason']})")
            else:
                ct = result.get("chunk_type_counts", {})
                parts = [f"{k}={v}" for k, v in sorted(ct.items())]
                print(
                    f"  {article_id}: OK  children={result['child_count']}"
                    f"  oversized={result['oversized']}"
                    f"  approx_spans={result['approximate_spans']}"
                    f"  {', '.join(parts)}"
                )
        except Exception as exc:
            print(f"  {article_id}: ERROR – {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
