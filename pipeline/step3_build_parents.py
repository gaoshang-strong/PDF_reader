#!/usr/bin/env python3
"""Step 3B-1: Build parent chunks from Step 3A structured blocks.

No LLM or external API. Text is never rewritten — parent text is formed
by joining source block texts with two newlines.

Reads:  data/index/{article_id}/structured_blocks.jsonl
Writes: data/index/{article_id}/parents.jsonl
        data/index/{article_id}/parent_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PARENT_TARGET_TOKENS = 1200
PARENT_MAX_TOKENS = 1800
PARENT_MIN_TOKENS = 200

BUILDER_VERSION = "3b1.1"

# block_types whose heading blocks (heading_level is not None) should be skipped
_SKIP_TYPES = {"heading", "title", "reference"}

# Caption block types → single-block parents, chunk_type preserved
_CAPTION_TYPES = {"figure_caption", "table_caption"}


# ---------------------------------------------------------------------------
# Token count
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# Parent factory
# ---------------------------------------------------------------------------

def _make_parent(
    article_id: str,
    parent_index: int,
    chunk_type: str,
    source_blocks: list[dict],
    oversized: bool = False,
) -> dict:
    text = "\n\n".join(b["text"] for b in source_blocks)
    section_path = source_blocks[0]["section_path"]
    char_start = source_blocks[0]["char_start"]
    char_end = source_blocks[-1]["char_end"]
    include_in_default_qa = chunk_type not in ("back_matter",)

    return {
        "parent_id": f"{article_id}::parent_{parent_index:06d}",
        "article_id": article_id,
        "parent_index": parent_index,
        "chunk_level": "parent",
        "chunk_type": chunk_type,
        "section_path": section_path,
        "source_block_ids": [b["block_id"] for b in source_blocks],
        "text": text,
        "char_start": char_start,
        "char_end": char_end,
        "token_count": _token_count(text),
        "oversized": oversized,
        "include_in_default_qa": include_in_default_qa,
    }


# ---------------------------------------------------------------------------
# Core grouping algorithm
# ---------------------------------------------------------------------------

def build_parents(
    blocks: list[dict],
    article_id: str,
    parent_max_tokens: int = PARENT_MAX_TOKENS,
) -> list[dict]:
    """Convert a flat list of structured blocks into parent chunks."""
    parents: list[dict] = []
    parent_index = 0

    # Running accumulator for body / back_matter paragraphs
    acc: list[dict] = []
    acc_tokens = 0
    acc_section_path: Optional[tuple] = None
    acc_chunk_type: Optional[str] = None

    def flush() -> None:
        nonlocal parent_index, acc, acc_tokens
        if not acc:
            return
        parents.append(_make_parent(article_id, parent_index, acc_chunk_type, acc))
        parent_index += 1
        acc = []
        acc_tokens = 0

    for block in blocks:
        btype = block["block_type"]
        hlevel = block["heading_level"]

        # --- skip: headings, titles, references, and back_matter headings ---
        if btype in _SKIP_TYPES:
            continue
        if btype == "back_matter" and hlevel is not None:
            # back_matter heading block — skip
            continue

        # --- captions: one block → one parent ---
        if btype in _CAPTION_TYPES:
            flush()
            parents.append(_make_parent(article_id, parent_index, btype, [block]))
            parent_index += 1
            continue

        # --- body paragraphs and back_matter content ---
        chunk_type = "body" if btype == "paragraph" else "back_matter"
        sp = tuple(block["section_path"])
        tok = block["token_count"]

        # Flush if section_path or chunk_type changes
        if sp != acc_section_path or chunk_type != acc_chunk_type:
            flush()
            acc_section_path = sp
            acc_chunk_type = chunk_type

        # Oversized single block
        if tok > parent_max_tokens:
            flush()
            parents.append(
                _make_parent(article_id, parent_index, chunk_type, [block], oversized=True)
            )
            parent_index += 1
            # Reset accumulator context (same section, ready for next block)
            acc_section_path = sp
            acc_chunk_type = chunk_type
            continue

        # Adding this block would exceed limit — flush first
        if acc and acc_tokens + tok > parent_max_tokens:
            flush()
            acc_section_path = sp
            acc_chunk_type = chunk_type

        acc.append(block)
        acc_tokens += tok

    flush()
    return parents


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(
    article_id: str,
    parents: list[dict],
    input_path: Path,
    output_path: Path,
    params: dict,
) -> dict:
    from collections import Counter
    type_counts = Counter(p["chunk_type"] for p in parents)
    oversized = sum(1 for p in parents if p.get("oversized"))
    return {
        "article_id": article_id,
        "builder_version": BUILDER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "parent_count": len(parents),
        "chunk_type_counts": dict(type_counts),
        "default_parameters": params,
        "oversized_parent_count": oversized,
    }


# ---------------------------------------------------------------------------
# Article processing
# ---------------------------------------------------------------------------

def process_article(
    article_id: str,
    data_root: Path,
    overwrite: bool = False,
    parent_target_tokens: int = PARENT_TARGET_TOKENS,
    parent_max_tokens: int = PARENT_MAX_TOKENS,
    parent_min_tokens: int = PARENT_MIN_TOKENS,
) -> dict:
    index_dir = data_root / "index" / article_id
    blocks_path = index_dir / "structured_blocks.jsonl"
    parents_path = index_dir / "parents.jsonl"
    manifest_path = index_dir / "parent_manifest.json"

    if not blocks_path.exists():
        return {"article_id": article_id, "skipped": True,
                "reason": f"structured_blocks.jsonl not found: {blocks_path}"}

    if parents_path.exists() and not overwrite:
        return {"article_id": article_id, "skipped": True, "reason": "output exists"}

    blocks = [
        json.loads(line)
        for line in blocks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Sort by block_index to guarantee document order
    blocks.sort(key=lambda b: b["block_index"])

    parents = build_parents(blocks, article_id, parent_max_tokens)

    # Write JSONL
    with parents_path.open("w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    params = {
        "parent_target_tokens": parent_target_tokens,
        "parent_max_tokens": parent_max_tokens,
        "parent_min_tokens": parent_min_tokens,
    }
    manifest = _build_manifest(article_id, parents, blocks_path, parents_path, params)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "article_id": article_id,
        "skipped": False,
        "parent_count": len(parents),
        "chunk_type_counts": manifest["chunk_type_counts"],
        "oversized": manifest["oversized_parent_count"],
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_article_ids(data_root: Path) -> list[str]:
    index_dir = data_root / "index"
    if not index_dir.exists():
        return []
    return sorted(
        d.name for d in index_dir.iterdir()
        if d.is_dir() and (d / "structured_blocks.jsonl").exists()
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3B-1: Build parent chunks from structured blocks (no LLM)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--article-id", help="Process a single article ID")
    parser.add_argument("--all", action="store_true", help="Process all indexed articles")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--parent-max-tokens", type=int, default=PARENT_MAX_TOKENS)
    parser.add_argument("--parent-target-tokens", type=int, default=PARENT_TARGET_TOKENS)
    parser.add_argument("--parent-min-tokens", type=int, default=PARENT_MIN_TOKENS)
    args = parser.parse_args()

    if args.article_id:
        article_ids = [args.article_id]
    else:
        article_ids = discover_article_ids(args.data_root)

    if not article_ids:
        print("No indexed articles found.")
        return

    print(f"Building parents for {len(article_ids)} article(s) ...")

    for article_id in article_ids:
        try:
            result = process_article(
                article_id,
                args.data_root,
                overwrite=args.overwrite,
                parent_target_tokens=args.parent_target_tokens,
                parent_max_tokens=args.parent_max_tokens,
                parent_min_tokens=args.parent_min_tokens,
            )
            if result.get("skipped"):
                print(f"  {article_id}: SKIPPED  ({result['reason']})")
            else:
                ct = result.get("chunk_type_counts", {})
                parts = [f"{k}={v}" for k, v in sorted(ct.items())]
                print(
                    f"  {article_id}: OK  parents={result['parent_count']}"
                    f"  oversized={result['oversized']}  {', '.join(parts)}"
                )
        except Exception as exc:
            print(f"  {article_id}: ERROR – {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
