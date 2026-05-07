#!/usr/bin/env python3
"""Audit structured_blocks.jsonl output from Step 3A.

Reads one or all articles from data/index/ and prints a structured report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

VERY_LONG_PARA_CHARS = 3000


def audit_article(article_id: str, data_root: Path) -> dict:
    index_dir = data_root / "index" / article_id
    jsonl_path = index_dir / "structured_blocks.jsonl"

    if not jsonl_path.exists():
        return {"article_id": article_id, "error": f"not found: {jsonl_path}"}

    blocks = []
    with jsonl_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return {"article_id": article_id, "error": f"JSON error line {lineno}: {exc}"}

    type_dist = Counter(b.get("block_type") for b in blocks)

    # Empty-text blocks
    empty_text = [b["block_id"] for b in blocks if not b.get("text", "").strip()]

    # Invalid char spans: char_start > char_end, or negative values
    bad_spans = [
        b["block_id"]
        for b in blocks
        if b.get("char_start", 0) > b.get("char_end", 0) or b.get("char_start", 0) < 0
    ]

    # Very long paragraphs
    long_paras = [
        b["block_id"]
        for b in blocks
        if b.get("block_type") == "paragraph" and len(b.get("text", "")) > VERY_LONG_PARA_CHARS
    ]

    # Body paragraphs missing section_path
    missing_section_path = [
        b["block_id"]
        for b in blocks
        if b.get("block_type") == "paragraph" and not b.get("section_path")
    ]

    return {
        "article_id": article_id,
        "total_blocks": len(blocks),
        "block_type_distribution": dict(type_dist),
        "heading_count": type_dist.get("heading", 0),
        "paragraph_count": type_dist.get("paragraph", 0),
        "figure_caption_count": type_dist.get("figure_caption", 0),
        "table_caption_count": type_dist.get("table_caption", 0),
        "reference_count": type_dist.get("reference", 0),
        "back_matter_count": type_dist.get("back_matter", 0),
        "title_count": type_dist.get("title", 0),
        "issues": {
            "empty_text_blocks": empty_text,
            "invalid_char_spans": bad_spans,
            "very_long_paragraphs": long_paras,
            "missing_section_path": missing_section_path,
        },
    }


def print_report(report: dict) -> None:
    aid = report["article_id"]
    if "error" in report:
        print(f"[{aid}] ERROR: {report['error']}")
        return

    print(f"\n{'='*60}")
    print(f"Article: {aid}")
    print(f"  Total blocks : {report['total_blocks']}")
    print(f"  Block types  :")
    for btype, count in sorted(report["block_type_distribution"].items()):
        print(f"    {btype:<25} {count}")

    issues = report["issues"]
    print(f"  Issues:")
    if issues["empty_text_blocks"]:
        print(f"    WARN  empty_text_blocks     : {len(issues['empty_text_blocks'])} block(s)")
        for bid in issues["empty_text_blocks"][:5]:
            print(f"      - {bid}")
    else:
        print(f"    OK    empty_text_blocks     : 0")

    if issues["invalid_char_spans"]:
        print(f"    WARN  invalid_char_spans    : {len(issues['invalid_char_spans'])} block(s)")
        for bid in issues["invalid_char_spans"][:5]:
            print(f"      - {bid}")
    else:
        print(f"    OK    invalid_char_spans    : 0")

    if issues["very_long_paragraphs"]:
        print(f"    INFO  very_long_paragraphs  : {len(issues['very_long_paragraphs'])} block(s) "
              f"(>{VERY_LONG_PARA_CHARS} chars)")
        for bid in issues["very_long_paragraphs"][:5]:
            print(f"      - {bid}")
    else:
        print(f"    OK    very_long_paragraphs  : 0")

    if issues["missing_section_path"]:
        print(f"    WARN  missing_section_path  : {len(issues['missing_section_path'])} block(s)")
        for bid in issues["missing_section_path"][:5]:
            print(f"      - {bid}")
    else:
        print(f"    OK    missing_section_path  : 0")


def discover_article_ids(data_root: Path) -> list[str]:
    index_dir = data_root / "index"
    if not index_dir.exists():
        return []
    return sorted(
        d.name for d in index_dir.iterdir()
        if d.is_dir() and (d / "structured_blocks.jsonl").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit structured_blocks.jsonl output from Step 3A."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory (default: ../data relative to this script)",
    )
    parser.add_argument("--article-id", help="Audit a single article ID")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output JSON instead of human-readable text")
    args = parser.parse_args()

    if args.article_id:
        article_ids = [args.article_id]
    else:
        article_ids = discover_article_ids(args.data_root)

    if not article_ids:
        print("No indexed articles found.", file=sys.stderr)
        sys.exit(1)

    reports = [audit_article(aid, args.data_root) for aid in article_ids]

    if args.json_output:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print_report(report)

        # Summary across all articles
        if len(reports) > 1:
            total_blocks = sum(r.get("total_blocks", 0) for r in reports if "error" not in r)
            total_issues = sum(
                len(r["issues"]["empty_text_blocks"])
                + len(r["issues"]["invalid_char_spans"])
                + len(r["issues"]["missing_section_path"])
                for r in reports
                if "error" not in r
            )
            print(f"\n{'='*60}")
            print(f"SUMMARY: {len(reports)} article(s)  |  total blocks={total_blocks}"
                  f"  |  total issues={total_issues}")


if __name__ == "__main__":
    main()
