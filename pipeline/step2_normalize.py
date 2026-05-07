#!/usr/bin/env python3
"""Step 2: Normalize raw article markdown using DeepSeek via OpenAI-compatible API."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

REQUIRED_HEADINGS = ["# Text", "# Images", "# Captions", "# Tables"]

SYSTEM_PROMPT = """\
You are a scientific article markdown formatter. Reorganize a raw MinerU-extracted \
markdown document into a clean, normalized structure.

Output exactly four top-level sections in this order, with no extra text before, \
between, or after them:

# Text
# Images
# Captions
# Tables

RULES — follow these exactly:

1. PRESERVE. Do not summarize, shorten, rewrite, or add anything not in the input.
2. # Text — Copy all prose, section headings, paragraphs, numbered/bulleted lists, \
inline equations, footnotes, and reference lists almost verbatim. This will be the \
largest section.
3. # Images — Move every markdown image link (syntax: ![alt](path)) to this section, \
one per line. Do not leave image links in # Text.
4. # Captions — Move obvious figure or table captions (lines that begin with "Figure", \
"Fig.", "Table", "Extended Data", "Supplementary", or similar label prefixes) to this \
section. Remove them from # Text.
5. # Tables — Move all markdown tables (consecutive lines that contain "|") to this \
section. Remove them from # Text.
6. If you are unsure where content belongs, leave it in # Text.
7. Your total output length must be very close to the input length. \
This is a reorganization, not a summary.
8. Output only the markdown. No preamble, no explanation, no code fences.

LATEX NORMALIZATION:

9. Convert obvious text-like LaTeX into normal readable text when you are confident \
it is just a word or abbreviation wrapped in LaTeX font commands. Examples:
   - $\\mathsf{TOX}$ → TOX
   - $\\mathtt{CD8}^{+}$ → CD8+
   - $\\mathtt{CD4}^{+}$ → CD4+
   - $\\left(\\mathsf{TOX}\\right)^{60-62}$ → TOX^60–62
   - $(\\mathsf{TLS})^{9,64,65}$ → TLS^9,64,65
   - $(\\mathsf{T}_{\\mathsf{PE}}\\mathsf{cells})$ → T_PE cells
   The test: if removing the LaTeX wrapper leaves a plain word, abbreviation, or \
   gene/protein name with optional superscript reference numbers, convert it.
10. Preserve real formulas as LaTeX. Do NOT convert equations, fractions, sums, \
integrals, statistical notation, complex scientific expressions, or any expression \
where conversion to plain text could change or obscure scientific meaning.
11. If you are uncertain whether a LaTeX span is text-like or formula-like, keep the \
original LaTeX unchanged.

INTERRUPTED SENTENCE REPAIR:

12. When the raw markdown clearly splits one natural sentence across a disruptive \
block — such as an image link, caption text, a markdown table, a display formula, \
a page/header/footer artifact, or a MinerU layout artifact — repair the sentence \
in # Text so it reads continuously.
13. Use context to decide whether adjacent text fragments belong to the same natural \
sentence. Any type of block can interrupt a sentence; do not assume only images do.
14. When an interrupting block is an image, caption, or table, move it to the \
appropriate section (Images / Captions / Tables) while keeping the repaired sentence \
continuous in # Text. Example:
     Raw:   "...predominance within the\\n\\n![](x.jpg)\\n\\nTME53, where they..."
     Text:  "...predominance within the TME53, where they..."
     Images: "![](x.jpg)"
15. When the interrupting block is a real display formula that belongs inside the \
sentence (e.g., an inline equation broken onto its own line by MinerU), preserve \
the formula in # Text at its natural position rather than deleting it.
16. If you are uncertain whether two text fragments should be joined, keep the \
original separation.
"""


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Return an OpenAI-compatible client pointed at DeepSeek."""
    from openai import OpenAI

    return OpenAI(
        api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL),
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def normalize_markdown(raw_md: str, client, model: str) -> str:
    """Call the LLM and return the normalized markdown string."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_md},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_normalized(raw_md: str, norm_md: str) -> dict:
    """
    Returns:
        ratio       – len(norm_md) / len(raw_md)
        missing_headings – list of REQUIRED_HEADINGS absent from norm_md
        status      – PASS | WARN | FAIL based on ratio
    """
    missing = [h for h in REQUIRED_HEADINGS if h not in norm_md]
    ratio = len(norm_md) / len(raw_md) if raw_md else 0.0

    if 0.95 <= ratio <= 1.05:
        status = "PASS"
    elif 0.90 <= ratio < 0.95 or 1.05 < ratio <= 1.10:
        status = "WARN"
    else:
        status = "FAIL"

    return {"ratio": round(ratio, 4), "missing_headings": missing, "status": status}


# ---------------------------------------------------------------------------
# Article processing
# ---------------------------------------------------------------------------

def process_article(
    article_id: str,
    data_root: Path,
    client,
    model: str = DEFAULT_MODEL,
    overwrite: bool = False,
) -> dict:
    """
    Normalize one article.  Returns a result dict with at minimum
    {"article_id": ..., "skipped": bool} or validation fields.
    """
    article_dir = data_root / "articles" / article_id
    raw_path = article_dir / "article_text.raw.md"
    norm_path = article_dir / "article_text.normalized.md"
    report_path = article_dir / "raw_article_report.json"

    if norm_path.exists() and not overwrite:
        return {"article_id": article_id, "skipped": True, "reason": "exists"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    raw_md = raw_path.read_text(encoding="utf-8")

    if report["length_status"] == "FAIL":
        norm_path.write_text(raw_md, encoding="utf-8")
        return {
            "article_id": article_id,
            "skipped": False,
            "llm_called": False,
            "reason": "FAIL length status – raw copied verbatim",
        }

    norm_md = normalize_markdown(raw_md, client, model)
    norm_path.write_text(norm_md, encoding="utf-8")

    validation = validate_normalized(raw_md, norm_md)
    return {"article_id": article_id, "skipped": False, "llm_called": True, **validation}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_article_ids(data_root: Path) -> list[str]:
    articles_dir = data_root / "articles"
    if not articles_dir.exists():
        return []
    return sorted(d.name for d in articles_dir.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Normalize article markdown using DeepSeek."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root data directory (default: ../data relative to this script)",
    )
    parser.add_argument("--article-id", help="Process a single article ID")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing normalized.md"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Silently skip articles that already have normalized.md (same as default)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model name")
    args = parser.parse_args()

    client = make_client()

    if args.article_id:
        article_ids = [args.article_id]
    else:
        article_ids = discover_article_ids(args.data_root)

    if not article_ids:
        print("No articles found.")
        return

    print(f"Normalizing {len(article_ids)} article(s) with model={args.model} ...")

    for article_id in article_ids:
        try:
            result = process_article(
                article_id, args.data_root, client, args.model, args.overwrite
            )
            if result.get("skipped"):
                print(f"  {article_id}: SKIPPED (normalized.md already exists)")
            elif not result.get("llm_called"):
                print(f"  {article_id}: COPIED RAW  ({result['reason']})")
            else:
                ratio = result.get("ratio", 0.0)
                status = result.get("status", "?")
                missing = result.get("missing_headings", [])
                line = f"  {article_id}: {status}  (ratio={ratio:.3f})"
                if missing:
                    line += f"  MISSING HEADINGS: {missing}"
                print(line)
        except Exception as exc:
            print(f"  {article_id}: ERROR – {exc}")


if __name__ == "__main__":
    main()
