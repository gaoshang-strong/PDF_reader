#!/usr/bin/env python3
"""Audit children.jsonl output from Step 3B-2.

Reports per-article statistics and flags issues.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CHILD_MAX_TOKENS = 550

VERY_SMALL_TOKENS = 30
VERY_LARGE_TOKENS = CHILD_MAX_TOKENS


def _median(vals: list) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def audit_article(article_id: str, data_root: Path) -> dict:
    index_dir = data_root / "index" / article_id
    children_path = index_dir / "children.jsonl"
    parents_path = index_dir / "parents.jsonl"

    if not children_path.exists():
        return {"article_id": article_id, "error": f"not found: {children_path}"}

    children = []
    with children_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                children.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return {"article_id": article_id, "error": f"JSON error line {lineno}: {exc}"}

    # Load parent ids for coverage check
    body_parent_ids: set[str] = set()
    if parents_path.exists():
        with parents_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    if p.get("chunk_type") == "body":
                        body_parent_ids.add(p["parent_id"])
                except json.JSONDecodeError:
                    pass

    type_counts = Counter(c["chunk_type"] for c in children)
    tokens = [c["token_count"] for c in children]

    # QA flag distribution
    qa_true = sum(1 for c in children if c.get("include_in_default_qa", False))
    qa_false = len(children) - qa_true

    # Issues
    empty = [c["chunk_id"] for c in children if not c.get("text", "").strip()]
    missing_parent_id = [c["chunk_id"] for c in children if not c.get("parent_id")]
    missing_section_path = [c["chunk_id"] for c in children if not c.get("section_path")]
    invalid_spans = [
        c["chunk_id"]
        for c in children
        if c.get("char_start", 0) > c.get("char_end", 0) or c.get("char_start", 0) < 0
    ]
    exceeds_max = [
        c["chunk_id"]
        for c in children
        if c["token_count"] > CHILD_MAX_TOKENS
    ]
    oversized = [c["chunk_id"] for c in children if c.get("oversized")]
    approx_spans = [c["chunk_id"] for c in children if c.get("span_is_approximate")]

    # Parent coverage: every body parent should have at least one child
    child_parent_ids = {c["parent_id"] for c in children}
    uncovered_body_parents = sorted(body_parent_ids - child_parent_ids)

    return {
        "article_id": article_id,
        "total_children": len(children),
        "chunk_type_counts": dict(type_counts),
        "token_stats": {
            "min": min(tokens) if tokens else 0,
            "median": _median(tokens),
            "max": max(tokens) if tokens else 0,
        },
        "oversized_count": len(oversized),
        "approximate_span_count": len(approx_spans),
        "qa_flag_distribution": {"include_in_default_qa_true": qa_true, "false": qa_false},
        "issues": {
            "empty_children": empty,
            "missing_parent_id": missing_parent_id,
            "missing_section_path": missing_section_path,
            "invalid_char_spans": invalid_spans,
            "children_exceeding_max_tokens": exceeds_max,
            "uncovered_body_parents": uncovered_body_parents,
        },
    }


def print_report(report: dict) -> None:
    aid = report["article_id"]
    if "error" in report:
        print(f"[{aid}] ERROR: {report['error']}")
        return

    ts = report["token_stats"]
    print(f"\n{'='*60}")
    print(f"Article: {aid}")
    print(f"  Total children : {report['total_children']}")
    print(f"  Chunk types    :")
    for k, v in sorted(report["chunk_type_counts"].items()):
        print(f"    {k:<25} {v}")
    print(
        f"  Token stats    : min={ts['min']}  median={ts['median']:.0f}  max={ts['max']}"
    )
    print(f"  Oversized      : {report['oversized_count']}")
    print(f"  Approx spans   : {report['approximate_span_count']}")
    qa = report["qa_flag_distribution"]
    print(
        f"  Default QA     : include=True:{qa['include_in_default_qa_true']}"
        f"  include=False:{qa['false']}"
    )

    issues = report["issues"]
    ok = True
    for key, label in [
        ("empty_children", "empty children"),
        ("missing_parent_id", "missing parent_id"),
        ("missing_section_path", "missing section_path"),
        ("invalid_char_spans", "invalid char spans"),
        ("children_exceeding_max_tokens", "children exceeding max tokens"),
        ("uncovered_body_parents", "body parents with no children"),
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
        description="Audit child chunks from Step 3B-2."
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
            d.name
            for d in index_dir.iterdir()
            if d.is_dir() and (d / "children.jsonl").exists()
        )

    if not article_ids:
        print("No child files found.", file=sys.stderr)
        sys.exit(1)

    reports = [audit_article(aid, args.data_root) for aid in article_ids]

    if args.json_output:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return

    for report in reports:
        print_report(report)

    if len(reports) > 1:
        total = sum(r.get("total_children", 0) for r in reports if "error" not in r)
        oversized = sum(r.get("oversized_count", 0) for r in reports if "error" not in r)
        approx = sum(r.get("approximate_span_count", 0) for r in reports if "error" not in r)
        total_issues = sum(
            sum(len(v) for v in r["issues"].values())
            for r in reports
            if "error" not in r
        )
        print(f"\n{'='*60}")
        print(
            f"SUMMARY: {len(reports)} article(s)"
            f"  |  total_children={total}"
            f"  |  oversized={oversized}"
            f"  |  approx_spans={approx}"
            f"  |  issues={total_issues}"
        )


if __name__ == "__main__":
    main()
