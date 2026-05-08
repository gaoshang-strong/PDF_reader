#!/usr/bin/env python3
"""Bug-fix script: patch metadata in article_card.json and re-render article_card.md.

Step 4C previously read metadata from data/working/{id}/metadata.json (wrong path)
and used lowercase field names doi/year instead of DOI/publication_year (wrong keys).
This script corrects the metadata block in-place without re-running LLM extraction.

For each article that has an article_card.json:
  1. Load the correct metadata from data/articles/{id}/metadata.json.
  2. Patch card["metadata"] with title, authors, journal, year, doi.
  3. Overwrite article_card.json.
  4. Re-render article_card.md.

Usage
-----
  python scripts/fix_article_card_metadata.py
  python scripts/fix_article_card_metadata.py --article-ids pdf_abc pdf_def
  python scripts/fix_article_card_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step4C_extract_article_card import build_metadata_block
from pipeline.step4D_render_article_card_md import render_article_card


def load_source_metadata(article_id: str, data_root: Path) -> dict | None:
    path = data_root / "articles" / article_id / "metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fix_one(article_id: str, data_root: Path, dry_run: bool) -> str:
    """Patch metadata and re-render for one article. Returns a status string."""
    card_dir = data_root / "literature_cards" / article_id
    card_path = card_dir / "article_card.json"
    md_path = card_dir / "article_card.md"

    if not card_path.exists():
        return "skip  (no article_card.json)"

    with open(card_path) as f:
        card = json.load(f)

    raw_meta = load_source_metadata(article_id, data_root)
    if raw_meta is None:
        return "skip  (no source metadata.json)"

    new_meta = build_metadata_block(raw_meta)
    old_title = card.get("metadata", {}).get("title")
    new_title = new_meta.get("title")

    if old_title == new_title and old_title is not None:
        return f"ok    (title already present: {old_title[:60]})"

    if not dry_run:
        card["metadata"] = new_meta
        with open(card_path, "w") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)

        md = render_article_card(card)
        md_path.write_text(md, encoding="utf-8")

    change = f"{str(old_title)[:30]!r} → {str(new_title)[:60]!r}"
    prefix = "dry   " if dry_run else "fixed "
    return f"{prefix}title: {change}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch article_card.json metadata and re-render article_card.md."
    )
    parser.add_argument("--article-ids", nargs="+", metavar="ID", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing files.")
    args = parser.parse_args()

    data_root = REPO_ROOT / "data"
    lit_dir = data_root / "literature_cards"

    if args.article_ids:
        article_ids = args.article_ids
    else:
        article_ids = sorted(d.name for d in lit_dir.iterdir() if d.is_dir()) if lit_dir.exists() else []

    if not article_ids:
        print("No article cards found.")
        sys.exit(0)

    print(f"{'DRY RUN — ' if args.dry_run else ''}Fixing metadata for {len(article_ids)} article(s)...\n")
    for article_id in article_ids:
        status = fix_one(article_id, data_root, dry_run=args.dry_run)
        print(f"  {article_id}  {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
