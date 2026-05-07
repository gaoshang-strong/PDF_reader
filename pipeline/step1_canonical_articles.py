#!/usr/bin/env python3
"""Step 1: Create canonical article folders from MinerU and GROBID outputs."""

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_article_ids(data_root: Path) -> list[str]:
    mineru_dir = data_root / "parser_outputs" / "mineru_work"
    if not mineru_dir.exists():
        return []
    return sorted(d.name for d in mineru_dir.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Copy operations
# ---------------------------------------------------------------------------

def copy_raw_markdown(article_id: str, data_root: Path, overwrite: bool = False) -> Path:
    src = data_root / "parser_outputs" / "mineru_work" / article_id / article_id / "full.md"
    dst = data_root / "articles" / article_id / "article_text.raw.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or overwrite:
        shutil.copy2(src, dst)
    return dst


def copy_images(article_id: str, data_root: Path, overwrite: bool = False) -> Optional[Path]:
    src = data_root / "parser_outputs" / "mineru_work" / article_id / article_id / "images"
    dst = data_root / "articles" / article_id / "images"
    if not src.exists():
        return None
    if dst.exists():
        if not overwrite:
            return dst
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


# ---------------------------------------------------------------------------
# GROBID TEI parsing helpers
# ---------------------------------------------------------------------------

def _all_text(el: ET.Element) -> str:
    return ET.tostring(el, method="text", encoding="unicode").strip()


def _find(el: ET.Element, path: str) -> Optional[ET.Element]:
    return el.find(path, _NS)


def _findall(el: ET.Element, path: str) -> list[ET.Element]:
    return el.findall(path, _NS)


def _findtext(el: ET.Element, path: str) -> Optional[str]:
    t = el.findtext(path, namespaces=_NS)
    return t.strip() if t and t.strip() else None


def _parse_author_name(author_el: ET.Element) -> Optional[str]:
    pn = _find(author_el, "tei:persName")
    if pn is None:
        return None
    first = _findtext(pn, ".//tei:forename[@type='first']") or _findtext(pn, "tei:forename")
    last = _findtext(pn, "tei:surname")
    name = " ".join(p for p in [first, last] if p)
    return name or None


def _parse_affiliation_text(aff_el: ET.Element) -> str:
    parts = []
    for org in _findall(aff_el, ".//tei:orgName"):
        if org.text and org.text.strip():
            parts.append(org.text.strip())
    for loc in ("tei:settlement", "tei:country"):
        t = _findtext(aff_el, f".//tei:{loc.split(':')[1]}")
        if t:
            parts.append(t)
    return ", ".join(parts)


def _parse_reference(bib: ET.Element) -> dict:
    ref: dict = {}
    title_el = _find(bib, ".//tei:analytic/tei:title[@type='main']") or _find(
        bib, ".//tei:analytic/tei:title"
    )
    if title_el is not None:
        t = _all_text(title_el)
        if t:
            ref["title"] = t
    journal_el = _find(bib, ".//tei:monogr/tei:title[@level='j']") or _find(
        bib, ".//tei:monogr/tei:title"
    )
    if journal_el is not None and journal_el.text and journal_el.text.strip():
        ref["journal"] = journal_el.text.strip()
    doi_el = _find(bib, ".//tei:idno[@type='DOI']")
    if doi_el is not None and doi_el.text and doi_el.text.strip():
        ref["doi"] = doi_el.text.strip()
    date_el = _find(bib, ".//tei:date[@type='published']") or _find(bib, ".//tei:date")
    if date_el is not None:
        when = date_el.get("when", "")
        if when and when[:4].isdigit():
            ref["year"] = int(when[:4])
    authors = [
        name
        for a in _findall(bib, ".//tei:author")
        if (name := _parse_author_name(a)) is not None
    ]
    if authors:
        ref["authors"] = authors
    return ref


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_grobid_metadata(article_id: str, data_root: Path) -> dict:
    tei_path = data_root / "parser_outputs" / "grobid_tei" / f"{article_id}.tei.xml"

    base: dict = {
        "article_id": article_id,
        "title": None,
        "authors": [],
        "affiliations": [],
        "journal": None,
        "DOI": None,
        "publication_year": None,
        "abstract": None,
        "keywords": [],
        "references": [],
    }

    if not tei_path.exists():
        return base

    try:
        tree = ET.parse(tei_path)
    except ET.ParseError:
        return base

    root = tree.getroot()

    # Title
    title_el = _find(root, ".//tei:titleStmt/tei:title[@type='main']")
    if title_el is not None:
        t = _all_text(title_el)
        if t:
            base["title"] = t

    source_desc = _find(root, ".//tei:sourceDesc")
    if source_desc is not None:
        # Authors (deduplicated, preserving order)
        seen: set[str] = set()
        for a in _findall(source_desc, ".//tei:author"):
            name = _parse_author_name(a)
            if name and name not in seen:
                seen.add(name)
                base["authors"].append(name)

        # Affiliations (unique by key, preserving first-seen order)
        aff_map: dict[str, str] = {}
        for a in _findall(source_desc, ".//tei:author"):
            for aff in _findall(a, "tei:affiliation"):
                key = aff.get("key", "")
                if key not in aff_map:
                    text = _parse_affiliation_text(aff)
                    if text:
                        aff_map[key] = text
        base["affiliations"] = list(aff_map.values())

        # DOI
        doi_el = _find(source_desc, ".//tei:idno[@type='DOI']")
        if doi_el is not None and doi_el.text and doi_el.text.strip():
            base["DOI"] = doi_el.text.strip()

        # Publication year
        date_el = _find(source_desc, ".//tei:date[@type='published']")
        if date_el is not None:
            when = date_el.get("when", "")
            if when and when[:4].isdigit():
                base["publication_year"] = int(when[:4])

        # Journal
        j_el = _find(source_desc, ".//tei:monogr/tei:title[@level='j']")
        if j_el is not None and j_el.text and j_el.text.strip():
            base["journal"] = j_el.text.strip()

    # Abstract
    ab_el = _find(root, ".//tei:abstract")
    if ab_el is not None:
        ab_text = _all_text(ab_el)
        if ab_text:
            base["abstract"] = ab_text

    # Keywords
    base["keywords"] = [
        kw.text.strip()
        for kw in _findall(root, ".//tei:keywords/tei:term")
        if kw.text and kw.text.strip()
    ]

    # References
    base["references"] = [
        ref
        for bib in _findall(root, ".//tei:back//tei:listBibl/tei:biblStruct")
        if (ref := _parse_reference(bib))
    ]

    return base


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def compute_report(article_id: str, data_root: Path) -> dict:
    text = (data_root / "articles" / article_id / "article_text.raw.md").read_text(
        encoding="utf-8"
    )
    char_count = len(text)
    word_count = len(text.split())
    token_count = math.ceil(char_count / 4)

    if token_count <= 90_000:
        status = "PASS"
    elif token_count <= 115_000:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "article_id": article_id,
        "raw_char_count": char_count,
        "raw_word_count": word_count,
        "estimated_token_count": token_count,
        "length_status": status,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_article(article_id: str, data_root: Path, overwrite: bool = False) -> dict:
    article_dir = data_root / "articles" / article_id
    article_dir.mkdir(parents=True, exist_ok=True)

    copy_raw_markdown(article_id, data_root, overwrite)
    copy_images(article_id, data_root, overwrite)

    meta_path = article_dir / "metadata.json"
    if not meta_path.exists() or overwrite:
        metadata = extract_grobid_metadata(article_id, data_root)
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    report = compute_report(article_id, data_root)
    report_path = article_dir / "raw_article_report.json"
    if not report_path.exists() or overwrite:
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Build canonical article folders from MinerU + GROBID outputs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory (default: ../data relative to this script)",
    )
    parser.add_argument("--article-id", help="Process a single article ID")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    args = parser.parse_args()

    data_root: Path = args.data_root

    if args.article_id:
        article_ids = [args.article_id]
    else:
        article_ids = discover_article_ids(data_root)

    if not article_ids:
        print("No articles found.")
        return

    print(f"Processing {len(article_ids)} article(s) in {data_root / 'articles'} ...")

    results = []
    errors = []
    for article_id in article_ids:
        try:
            report = process_article(article_id, data_root, args.overwrite)
            results.append(report)
            print(
                f"  {article_id}: {report['length_status']}"
                f" ({report['estimated_token_count']:,} est. tokens)"
            )
        except Exception as exc:
            errors.append((article_id, exc))
            print(f"  {article_id}: ERROR – {exc}")

    print(f"\nDone. {len(results)}/{len(article_ids)} articles processed.")
    if results:
        counts = {s: sum(1 for r in results if r["length_status"] == s) for s in ("PASS", "WARN", "FAIL")}
        print(f"  PASS: {counts['PASS']}  WARN: {counts['WARN']}  FAIL: {counts['FAIL']}")
    if errors:
        print(f"  Errors: {len(errors)}")


if __name__ == "__main__":
    main()
