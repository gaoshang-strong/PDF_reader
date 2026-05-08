# Step 3A: Structure Normalized Markdown

## Goal

Convert the normalized Markdown produced by Step 2 into a flat, ordered
sequence of typed content blocks stored in JSONL. Each block carries its
type, section hierarchy, character offsets, and a token estimate.

This structured representation is the foundation for all downstream RAG
chunking: Steps 3B-1 and 3B-2 consume it to build parent and child chunks.

---

## Key Idea

All parsing is **deterministic and LLM-free** — the same input always
produces the same blocks.

- **Reproducibility**: block IDs are stable references across runs.
- **Auditability**: every classification decision traces to a regex or
  line pattern, not a black box.
- **Speed**: parsing an article takes milliseconds with zero API cost.

The parser reads line by line: blank lines mark boundaries, `#` lines
update the section path, and content lines are classified by regex
(caption patterns, reference numbering, back-matter headings).

---

## Inputs

| Source | Path |
|---|---|
| Normalized Markdown | `data/articles/{article_id}/article_text.normalized.md` |

---

## Outputs

| File | Path | Description |
|---|---|---|
| Structured blocks | `data/index/{article_id}/structured_blocks.jsonl` | One JSON object per line, one block per object |
| Manifest | `data/index/{article_id}/structure_manifest.json` | Block counts and metadata |

### Block Schema

```json
{
  "block_id": "pdf_abc_000003",
  "article_id": "pdf_abc",
  "block_index": 3,
  "block_type": "paragraph",
  "section_path": ["Text", "Methods", "Statistics"],
  "heading_level": null,
  "text": "Trimmed text of the block...",
  "char_start": 1420,
  "char_end": 1650,
  "token_count": 42
}
```

### Block Types

| Type | Meaning |
|---|---|
| `title` | First non-artifact heading in `# Text` that looks like an article title |
| `heading` | Markdown heading (`#`, `##`, ...) |
| `paragraph` | General prose — the most common type |
| `figure_caption` | Caption starting with `Figure N.` or `Fig. N.` |
| `table_caption` | Caption starting with `Table N.` |
| `reference` | Individual numbered reference entry under a REFERENCES heading |
| `back_matter` | Content under ACKNOWLEDGMENTS, AUTHOR CONTRIBUTIONS, etc. |

---

## Parsing Rules

1. **Blank lines** mark block boundaries.
2. **Headings** (`# text`) update `section_path` by level.
3. **Consecutive content lines** (no blank line) are merged into one paragraph block.
4. **References mode** is entered on headings matching `REFERENCES / REFERENCE LIST / BIBLIOGRAPHY`; lines matching `^\d+\.\s` start a new reference block.
5. **Back-matter mode** is entered on headings matching `ACKNOWLEDGMENTS`, `AUTHOR CONTRIBUTIONS`, `COMPETING INTERESTS`, `FUNDING`, etc.
6. **Figure captions** — first line matches `^(Figure|Fig\.?)\s+\d+`.
7. **Table captions** — first line matches `^Table\s+\d+`.
8. **Title detection** — first L1 heading in `# Text`, ≥ 20 chars, not all-caps.
9. **Image links** inside `# Images` are silently skipped.
10. Text is **never rewritten** — only leading/trailing whitespace is stripped.

---

## How to Run

```bash
# Single article
python pipeline/step3_structure_normalized.py --article-id pdf_16edbbde296287d6

# All articles
python pipeline/step3_structure_normalized.py --all

# Overwrite existing output
python pipeline/step3_structure_normalized.py --all --overwrite
```

---

## Python API

```python
from pathlib import Path
from pipeline.step3_structure_normalized import parse_normalized_md, process_article

# Parse a markdown string directly
blocks = parse_normalized_md(text, article_id="pdf_abc")

# Process a full article (reads file, writes JSONL)
result = process_article("pdf_abc", data_root=Path("data"), overwrite=True)
```

---

## Known Limitations

- References without an explicit REFERENCES heading are classified as `paragraph`.
- Token count uses word count, not subword tokenization.
- Heading levels deeper than the article structure (e.g. `####`) parse correctly but produce deep `section_path` values.
