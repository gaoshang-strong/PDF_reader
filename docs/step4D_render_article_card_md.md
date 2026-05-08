# Step 4D: Render Article Card as Markdown

## Goal

Read an `article_card.json` and render it as a human-readable Markdown
document for quick review in a text editor, documentation tool, or GitHub
preview.

---

## Key Idea

Each of the nine modules has a dedicated renderer that picks the right
presentation for its content type: tables for structured lists (cohorts,
datasets, methods, pipeline steps), bullet lists for findings and
limitations, and prose for research question and mechanism model. Every
module section ends with an **Evidence parents** line listing the short IDs
of all parent chunks cited anywhere in that module (deduplicated, prefix
stripped to show just `parent_000001`).

---

## Inputs

| Source | Path |
|---|---|
| Article card JSON | `data/literature_cards/{article_id}/article_card.json` |

---

## Outputs

| File | Path |
|---|---|
| Rendered Markdown | `data/literature_cards/{article_id}/article_card.md` |

### Document Structure

```
# {article_id}
**Title:** ...
**Authors:** ...
**Journal:** ... · {year}
**DOI:** ...
**Extracted:** {timestamp}  model: {model}
---

## A. Research Question
{summary}
**Disease context:** ...
**Evidence parents:** parent_000001, parent_000003
---

## B. Study Design
...

... (nine module sections) ...

## Extraction Warnings          ← only if warnings present
- ...
```

### Module Rendering

| Module | Rendered as |
|---|---|
| A. Research Question | Text fields: disease/biological context, main question, stated gap, study type |
| B. Study Design | Text fields + table of comparison groups (Group A / Group B / Purpose) |
| C. Samples & Cohorts | Table: Cohort · Species · Disease · N · Material · Used for |
| D. Data Sources & Provenance | Table: Dataset · Modality · Source type · Accession |
| E. Omics & Experimental Methods | Table: Method · Category · Material · Purpose |
| F. Computational Pipeline | Table: Step · Software / Method · Parameters |
| G. Key Findings | Bullet list; `result_type` appended as italic tag |
| H. Mechanism Model | Per-claim: chain, upstream, mediators, downstream, cell types, pathways, implication |
| I. Limitations & Future Directions | Bullet list with `[limitation]` / `[caveat]` / `[future_direction]` tags |

---

## How to Run

```bash
# Render article card for one article
python pipeline/step4D_render_article_card_md.py --article-id pdf_b984e5bc1768479a

# Overwrite existing article_card.md
python pipeline/step4D_render_article_card_md.py --article-id pdf_b984e5bc1768479a --overwrite
```

---

## Python API

```python
from pathlib import Path
from pipeline.step4D_render_article_card_md import render_article_card, render_article_card_file

# Render from a dict (e.g. returned by extract_article_card)
md_string = render_article_card(card_dict)

# Read JSON, write .md, return output path
md_path = render_article_card_file("pdf_b984e5bc1768479a", overwrite=True)

# Custom data root (for testing)
md_path = render_article_card_file(
    "pdf_test",
    overwrite=True,
    data_root=Path("/tmp/test_data"),
)
```

---

## Evidence Parents Line

At the end of each module section, all unique `parent_id` values cited
anywhere in that module (including nested inside `items`, `claims`, and
`comparison_groups`) are listed with the `pdf_xxx::` prefix stripped:

```
**Evidence parents:** parent_000003, parent_000007, parent_000012
```

If no evidence was cited:

```
_Evidence parents: none_
```
