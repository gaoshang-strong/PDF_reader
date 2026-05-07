#!/usr/bin/env python3
"""Step 3A: Deterministic structure parser — converts normalized markdown into structured blocks.

No LLM or external API is used. Text is preserved exactly (only leading/trailing
whitespace is trimmed per block). char_start / char_end are byte offsets into the
original normalized.md string.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARSER_VERSION = "3a.1"

_STEP2_TOP_SECTIONS = {"Text", "Images", "Captions", "Tables"}

_BACK_MATTER_RE = re.compile(
    r"^(ACKNOWLEDGMENTS?|ACKNOWLEDGEMENTS?|AUTHOR\s+CONTRIBUTIONS?"
    r"|DECLARATION\s+OF\s+(COMPETING\s+)?INTERESTS?"
    r"|SUPPLEMENTAL\s+(INFORMATION|DATA)"
    r"|DATA\s+AND\s+CODE\s+AVAILABILITY"
    r"|COMPETING\s+INTERESTS?"
    r"|CONFLICT\s+OF\s+INTEREST"
    r"|FUNDING)$",
    re.IGNORECASE,
)

_REFERENCES_RE = re.compile(
    r"^(REFERENCES?|REFERENCE\s+LIST|BIBLIOGRAPHY|ONLINE\s+CONTENT)$",
    re.IGNORECASE,
)

_FIGURE_CAPTION_RE = re.compile(r"^(Figure|Fig\.?)\s+\d+", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+", re.IGNORECASE)
_REFERENCE_ITEM_RE = re.compile(r"^\d+\.\s")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


# ---------------------------------------------------------------------------
# Token count approximation
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Rough token estimate using word count."""
    return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# Line offset index
# ---------------------------------------------------------------------------

def _build_line_offsets(text: str) -> list[int]:
    """Return the character offset of the start of each line in text."""
    offsets: list[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(line)
    # Handle trailing newline-less text
    return offsets


# ---------------------------------------------------------------------------
# Block factory
# ---------------------------------------------------------------------------

def _make_block(
    article_id: str,
    block_index: int,
    block_type: str,
    section_path: list[str],
    heading_level: Optional[int],
    text: str,
    char_start: int,
    char_end: int,
) -> dict:
    return {
        "block_id": f"{article_id}_{block_index:06d}",
        "article_id": article_id,
        "block_index": block_index,
        "block_type": block_type,
        "section_path": list(section_path),
        "heading_level": heading_level,
        "text": text.strip(),
        "char_start": char_start,
        "char_end": char_end,
        "token_count": _token_count(text.strip()),
    }


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def _section_path_for_level(current: list[str], level: int, heading_text: str) -> list[str]:
    """Return updated section_path after encountering a heading at the given level."""
    base = current[: level - 1]
    return base + [heading_text]


def parse_normalized_md(text: str, article_id: str) -> list[dict]:  # noqa: C901
    """Parse normalized markdown into a flat list of structured block dicts."""
    lines = text.splitlines()
    offsets = _build_line_offsets(text)

    blocks: list[dict] = []
    block_index = 0

    section_path: list[str] = []
    in_references = False
    in_back_matter = False
    title_found = False

    # We skip lines from # Images section (image links) to keep blocks clean.
    # We do process # Captions (figure/table captions) and # Tables (as paragraphs).
    in_images_section = False

    i = 0
    n = len(lines)

    def emit(block_type, heading_level, text_str, first_i, last_i):
        nonlocal block_index
        cs = offsets[first_i]
        ce = offsets[last_i] + len(lines[last_i])
        b = _make_block(article_id, block_index, block_type, section_path,
                        heading_level, text_str, cs, ce)
        blocks.append(b)
        block_index += 1

    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        # --- skip blank lines ---
        if not stripped:
            i += 1
            continue

        # --- heading line ---
        m = _HEADING_RE.match(stripped)
        if m:
            hashes, heading_text = m.group(1), m.group(2).strip()
            level = len(hashes)

            # Update section_path
            section_path = _section_path_for_level(section_path, level, heading_text)

            # Detect state transitions
            heading_upper = heading_text.upper().strip()

            if heading_text in _STEP2_TOP_SECTIONS and level == 1:
                # Top-level Step 2 artifact sections
                in_images_section = (heading_text == "Images")
                in_references = False
                in_back_matter = False
                btype = "heading"

            elif _REFERENCES_RE.match(heading_upper):
                in_references = True
                in_back_matter = False
                in_images_section = False
                btype = "heading"

            elif _BACK_MATTER_RE.match(heading_upper):
                in_back_matter = True
                in_references = False
                in_images_section = False
                btype = "back_matter"

            else:
                # If we hit a heading after references/back_matter it may reset state,
                # but only if it's at level 1 (a new major section).
                if level == 1:
                    in_references = False
                    in_back_matter = False
                in_images_section = False

                # Title detection: first real heading in Text section, not all-caps,
                # at level 1, >= 20 chars.
                if (
                    not title_found
                    and level == 1
                    and heading_text not in _STEP2_TOP_SECTIONS
                    and len(heading_text) >= 20
                    and heading_text != heading_text.upper()
                ):
                    btype = "title"
                    title_found = True
                elif in_back_matter:
                    btype = "back_matter"
                else:
                    btype = "heading"

            emit(btype, level, heading_text, i, i)
            i += 1
            continue

        # --- content line (not a heading) ---

        # Skip lines inside # Images section (image link syntax)
        if in_images_section:
            i += 1
            continue

        # Accumulate a content group: consecutive non-heading non-empty lines.
        # Split the group whenever a figure/table caption start is encountered
        # mid-group (even without a preceding blank line).
        group_start = i
        group_lines: list[str] = []
        while i < n and lines[i].strip() and not _HEADING_RE.match(lines[i].strip()):
            group_lines.append(lines[i])
            i += 1

        if not group_lines:
            i += 1
            continue

        # Split the accumulated group into sub-groups at caption boundaries.
        # A new sub-group starts when a line (other than the very first) matches
        # a figure or table caption pattern.
        sub_groups: list[tuple[int, list[str]]] = []  # (abs_start_i, lines)
        cur_start = group_start
        cur_lines: list[str] = []
        for j, line in enumerate(group_lines):
            abs_j = group_start + j
            if j > 0 and (
                _FIGURE_CAPTION_RE.match(line.strip())
                or _TABLE_CAPTION_RE.match(line.strip())
            ):
                if cur_lines:
                    sub_groups.append((cur_start, cur_lines))
                cur_start = abs_j
                cur_lines = [line]
            else:
                cur_lines.append(line)
        if cur_lines:
            sub_groups.append((cur_start, cur_lines))

        for sub_start, sub_lines in sub_groups:
            first_stripped = sub_lines[0].strip()
            last_i = sub_start + len(sub_lines) - 1

            # --- reference mode: split sub-group into individual references ---
            if in_references:
                _emit_references(sub_lines, sub_start, article_id, section_path,
                                 offsets, lines, blocks)
                block_index = blocks[-1]["block_index"] + 1 if blocks else block_index
                continue

            # --- back matter mode ---
            if in_back_matter:
                text_str = " ".join(l.strip() for l in sub_lines)
                emit("back_matter", None, text_str, sub_start, last_i)
                continue

            # --- figure caption ---
            if _FIGURE_CAPTION_RE.match(first_stripped):
                text_str = " ".join(l.strip() for l in sub_lines)
                emit("figure_caption", None, text_str, sub_start, last_i)
                continue

            # --- table caption ---
            if _TABLE_CAPTION_RE.match(first_stripped):
                text_str = " ".join(l.strip() for l in sub_lines)
                emit("table_caption", None, text_str, sub_start, last_i)
                continue

            # --- regular paragraph ---
            text_str = " ".join(l.strip() for l in sub_lines)
            emit("paragraph", None, text_str, sub_start, last_i)

    return blocks


def _emit_references(
    group_lines: list[str],
    group_start: int,
    article_id: str,
    section_path: list[str],
    offsets: list[int],
    all_lines: list[str],
    blocks: list[dict],
) -> None:
    """Split a content group into individual reference blocks."""
    block_index = blocks[-1]["block_index"] + 1 if blocks else 0

    current_ref_lines: list[str] = []
    current_ref_start: Optional[int] = None

    def flush(ref_lines, ref_start_i):
        nonlocal block_index
        if not ref_lines:
            return
        text_str = " ".join(l.strip() for l in ref_lines).strip()
        if not text_str:
            return
        last_i = ref_start_i + len(ref_lines) - 1
        cs = offsets[ref_start_i]
        ce = offsets[last_i] + len(all_lines[last_i])
        b = _make_block(article_id, block_index, "reference", section_path,
                        None, text_str, cs, ce)
        blocks.append(b)
        block_index += 1

    for j, line in enumerate(group_lines):
        abs_i = group_start + j
        stripped = line.strip()

        if _REFERENCE_ITEM_RE.match(stripped):
            flush(current_ref_lines, current_ref_start)
            current_ref_lines = [line]
            current_ref_start = abs_i
        else:
            if current_ref_lines is not None and current_ref_start is not None:
                current_ref_lines.append(line)
            # else: text before first numbered item — treat as continuation paragraph
            # (edge case; skip for now)

    flush(current_ref_lines, current_ref_start)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(article_id: str, blocks: list[dict],
                    input_path: Path, output_dir: Path) -> dict:
    from collections import Counter
    type_dist = Counter(b["block_type"] for b in blocks)
    return {
        "article_id": article_id,
        "parser_version": PARSER_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_jsonl": str(output_dir / "structured_blocks.jsonl"),
        "block_count": len(blocks),
        "block_type_distribution": dict(type_dist),
        "section_count": sum(1 for b in blocks if b["block_type"] == "heading"),
        "paragraph_count": type_dist.get("paragraph", 0),
        "reference_count": type_dist.get("reference", 0),
        "figure_caption_count": type_dist.get("figure_caption", 0),
        "table_caption_count": type_dist.get("table_caption", 0),
    }


# ---------------------------------------------------------------------------
# Article processing
# ---------------------------------------------------------------------------

def process_article(
    article_id: str,
    data_root: Path,
    overwrite: bool = False,
) -> dict:
    """Parse one article. Returns a result dict."""
    norm_path = data_root / "articles" / article_id / "article_text.normalized.md"
    if not norm_path.exists():
        return {"article_id": article_id, "skipped": True,
                "reason": f"normalized.md not found: {norm_path}"}

    index_dir = data_root / "index" / article_id
    jsonl_path = index_dir / "structured_blocks.jsonl"
    manifest_path = index_dir / "structure_manifest.json"

    if jsonl_path.exists() and not overwrite:
        return {"article_id": article_id, "skipped": True, "reason": "output exists"}

    index_dir.mkdir(parents=True, exist_ok=True)

    text = norm_path.read_text(encoding="utf-8")
    blocks = parse_normalized_md(text, article_id)

    # Write JSONL
    with jsonl_path.open("w", encoding="utf-8") as f:
        for block in blocks:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")

    # Write manifest
    manifest = _build_manifest(article_id, blocks, norm_path, index_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    return {
        "article_id": article_id,
        "skipped": False,
        "block_count": len(blocks),
        "block_type_distribution": manifest["block_type_distribution"],
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_article_ids(data_root: Path) -> list[str]:
    articles_dir = data_root / "articles"
    if not articles_dir.exists():
        return []
    return sorted(
        d.name for d in articles_dir.iterdir()
        if d.is_dir() and (d / "article_text.normalized.md").exists()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3A: Parse normalized markdown into structured blocks (no LLM)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory (default: ../data relative to this script)",
    )
    parser.add_argument("--article-id", help="Process a single article ID")
    parser.add_argument("--all", action="store_true",
                        help="Process all articles with a normalized.md")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    if args.article_id:
        article_ids = [args.article_id]
    elif args.all:
        article_ids = discover_article_ids(args.data_root)
    else:
        article_ids = discover_article_ids(args.data_root)

    if not article_ids:
        print("No articles found.")
        return

    print(f"Structuring {len(article_ids)} article(s) ...")

    for article_id in article_ids:
        try:
            result = process_article(article_id, args.data_root, args.overwrite)
            if result.get("skipped"):
                print(f"  {article_id}: SKIPPED  ({result['reason']})")
            else:
                dist = result.get("block_type_distribution", {})
                parts = [f"{k}={v}" for k, v in sorted(dist.items())]
                print(f"  {article_id}: OK  blocks={result['block_count']}  {', '.join(parts)}")
        except Exception as exc:
            print(f"  {article_id}: ERROR – {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
