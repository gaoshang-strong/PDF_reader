# Step 1: Canonical Articles

## Goal

Ingest raw parser outputs (MinerU markdown + GROBID TEI XML) and produce a
clean, stable article folder for each paper. This is the single entry point
into the pipeline: every downstream step reads from `data/articles/`.

---

## Key Idea

Two tools parse the original PDF independently:

- **MinerU** extracts the full article text as raw Markdown, preserving
  layout structure (headings, paragraphs, figure/table captions).
- **GROBID** parses the PDF into TEI XML and provides reliable bibliographic
  metadata (title, authors, journal, year, DOI, abstract).

Step 1 does no LLM calls. It copies the MinerU Markdown verbatim, extracts
structured metadata from the GROBID XML, and produces a length report that
flags articles too long for the LLM normalization step.

**Three-tier length classification** (based on estimated token count from
character count ÷ 4):

| Status | Token range | Effect on Step 2 |
|---|---|---|
| PASS | ≤ 90,000 | LLM normalization runs normally |
| WARN | 90,001 – 115,000 | LLM normalization runs; manual review recommended |
| FAIL | > 115,000 | LLM call skipped; raw Markdown copied verbatim |

---

## Inputs

| Source | Path |
|---|---|
| MinerU raw Markdown | `data/parser_outputs/mineru_work/{article_id}/{article_id}/full.md` |
| MinerU images (optional) | `data/parser_outputs/mineru_work/{article_id}/{article_id}/images/` |
| GROBID TEI XML | `data/parser_outputs/grobid_tei/{article_id}.tei.xml` |

---

## Outputs

| File | Path | Description |
|---|---|---|
| Raw Markdown | `data/articles/{article_id}/article_text.raw.md` | Verbatim MinerU output |
| Metadata | `data/articles/{article_id}/metadata.json` | Title, authors, journal, year, DOI, abstract, affiliations |
| Length report | `data/articles/{article_id}/raw_article_report.json` | Token estimate, status (PASS/WARN/FAIL), char count |
| Images (optional) | `data/articles/{article_id}/images/` | Copied from MinerU output if present |

### `metadata.json` fields

| Field | Type | Description |
|---|---|---|
| `title` | string | Article title from GROBID |
| `authors` | list[string] | Deduplicated, order-preserving author names |
| `journal` | string | Journal name |
| `year` | int | Publication year |
| `doi` | string | DOI |
| `abstract` | string | Abstract text |
| `affiliations` | list[string] | Deduplicated institutional affiliations |

---

## How to Run

```bash
# All articles in data/parser_outputs/mineru_work/
python pipeline/step1_canonical_articles.py

# Single article
python pipeline/step1_canonical_articles.py --article-id pdf_b984e5bc1768479a

# Re-process already-processed articles
python pipeline/step1_canonical_articles.py --overwrite

# Custom data root
python pipeline/step1_canonical_articles.py --data-root /path/to/data
```

---

## Python API

```python
from pathlib import Path
from pipeline.step1_canonical_articles import process_article, discover_article_ids

# Discover all article IDs
ids = discover_article_ids(Path("data"))

# Process one article
report = process_article("pdf_b984e5bc1768479a", data_root=Path("data"), overwrite=True)
# report = {"article_id": ..., "status": "PASS", "token_estimate": 28145, ...}
```
