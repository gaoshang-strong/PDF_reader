#!/usr/bin/env python3
"""Audit parents.jsonl output from Step 3B-1.

Reports per-article statistics and flags issues.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

VERY_SMALL_TOKENS = 50
VERY_LARGE_TOKENS = 2000


def _median(vals: list) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def audit_article(article_id: str, data_root: Path) -> dict:
    index_dir = data_root / "index" / article_id
    parents_path = index_dir / "parents.jsonl"

    if not parents_path.exists():
        return {"article_id": article_id, "error": f"not found: {parents_path}"}

    parents = []
    with parents_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return {"article_id": article_id, "error": f"JSON error line {lineno}: {exc}"}

    type_counts = Counter(p["chunk_type"] for p in parents)
    tokens = [p["token_count"] for p in parents]

    empty_parents = [p["parent_id"] for p in parents if not p.get("text", "").strip()]
    invalid_spans = [
        p["parent_id"]
        for p in parents
        if p.get("char_start", 0) > p.get("char_end", 0) or p.get("char_start", 0) < 0
    ]
    missing_section_path = [
        p["parent_id"] for p in parents if not p.get("section_path")
    ]
    oversized = [p["parent_id"] for p in parents if p.get("oversized")]

    body_no_qa = [
        p["parent_id"]
        for p in parents
        if p["chunk_type"] == "body" and not p.get("include_in_default_qa", True)
    ]
    non_body_qa = [
        p["parent_id"]
        for p in parents
        if p["chunk_type"] not in ("body", "figure_caption", "table_caption")
        and p.get("include_in_default_qa", False)
    ]

    # Section coverage: unique section_paths that have at least one body parent
    body_sections = sorted(
        set(tuple(p["section_path"]) for p in parents if p["chunk_type"] == "body")
    )

    return {
        "article_id": article_id,
        "total_parents": len(parents),
        "chunk_type_counts": dict(type_counts),
        "token_stats": {
            "min": min(tokens) if tokens else 0,
            "median": _median(tokens),
            "max": max(tokens) if tokens else 0,
        },
        "oversized_count": len(oversized),
        "issues": {
            "empty_parents": empty_parents,
            "invalid_char_spans": invalid_spans,
            "missing_section_path": missing_section_path,
            "body_parents_excluded_from_qa": body_no_qa,
            "non_body_non_caption_parents_included_in_qa": non_body_qa,
        },
        "section_coverage": [list(sp) for sp in body_sections],
    }


def print_report(report: dict) -> None:
    aid = report["article_id"]
    if "error" in report:
        print(f"[{aid}] ERROR: {report['error']}")
        return

    ts = report["token_stats"]
    print(f"\n{'='*60}")
    print(f"Article: {aid}")
    print(f"  Total parents  : {report['total_parents']}")
    print(f"  Chunk types    :")
    for k, v in sorted(report["chunk_type_counts"].items()):
        print(f"    {k:<25} {v}")
    print(f"  Token stats    : min={ts['min']}  median={ts['median']:.0f}  max={ts['max']}")
    print(f"  Oversized      : {report['oversized_count']}")
    print(f"  Body sections  : {len(report['section_coverage'])} unique section path(s)")

    issues = report["issues"]
    ok = True
    for key, label in [
        ("empty_parents", "empty parents"),
        ("invalid_char_spans", "invalid char spans"),
        ("missing_section_path", "missing section_path"),
        ("body_parents_excluded_from_qa", "body parents excluded from QA"),
        ("non_body_non_caption_parents_included_in_qa", "unexpected include_in_default_qa=True"),
    ]:
        items = issues[key]
        if items:
            ok = False
            print(f"  WARN  {label:<40}: {len(items)}")
            for pid in items[:3]:
                print(f"    - {pid}")
        else:
            print(f"  OK    {label:<40}: 0")

    if ok:
        print("  Status: CLEAN")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit parent chunks from Step 3B-1."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--article-id", help="Audit a single article ID")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    if args.article_id:
        article_ids = [args.article_id]
    else:
        index_dir = args.data_root / "index"
        if not index_dir.exists():
            print("No index directory found.", file=sys.stderr)
            sys.exit(1)
        article_ids = sorted(
            d.name for d in index_dir.iterdir()
            if d.is_dir() and (d / "parents.jsonl").exists()
        )

    if not article_ids:
        print("No parent files found.", file=sys.stderr)
        sys.exit(1)

    reports = [audit_article(aid, args.data_root) for aid in article_ids]

    if args.json_output:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return

    for report in reports:
        print_report(report)

    if len(reports) > 1:
        total = sum(r.get("total_parents", 0) for r in reports if "error" not in r)
        oversized = sum(r.get("oversized_count", 0) for r in reports if "error" not in r)
        total_issues = sum(
            sum(len(v) for v in r["issues"].values())
            for r in reports if "error" not in r
        )
        print(f"\n{'='*60}")
        print(
            f"SUMMARY: {len(reports)} article(s)  |  total_parents={total}"
            f"  |  oversized={oversized}  |  issues={total_issues}"
        )


if __name__ == "__main__":
    main()
