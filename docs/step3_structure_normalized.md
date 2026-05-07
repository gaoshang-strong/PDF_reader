# Step 3A: Structure Normalized Markdown

## Purpose

Step 3A converts the normalized markdown produced by Step 2 into a flat sequence
of typed, indexed content blocks stored in JSONL format. Each block carries:
- Its type (`paragraph`, `heading`, `figure_caption`, `table_caption`, `reference`,
  `back_matter`, or `title`)
- The section hierarchy at that point (`section_path`)
- Character offsets into the original normalized markdown string
- A rough token count

This step provides the foundation for RAG chunking (Step 4). By structuring the
document now — without any LLM — later steps can make retrieval decisions using
stable, reproducible block boundaries.

## Input / Output

| | Path |
|---|---|
| Input | `data/articles/{article_id}/article_text.normalized.md` |
| Output JSONL | `data/index/{article_id}/structured_blocks.jsonl` |
| Output manifest | `data/index/{article_id}/structure_manifest.json` |

## Block Schema

Each line in `structured_blocks.jsonl` is a JSON object with these fields:

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

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `block_id` | string | `{article_id}_{block_index:06d}` — globally unique |
| `article_id` | string | Article folder name |
| `block_index` | int | 0-based sequential index within the article |
| `block_type` | string | One of the types listed below |
| `section_path` | list[str] | Heading hierarchy at this block's position |
| `heading_level` | int or null | 1–6 for heading blocks; null for content blocks |
| `text` | string | Block text, leading/trailing whitespace stripped |
| `char_start` | int | Byte offset of block start in the normalized.md string |
| `char_end` | int | Byte offset of block end (exclusive) in normalized.md |
| `token_count` | int | Word-count approximation for token budgeting |

### Block Types

| Type | Meaning |
|---|---|
| `title` | First non-artifact heading that looks like an article title |
| `heading` | Markdown heading (`#`, `##`, `###`, ...) |
| `paragraph` | General prose — the most common type |
| `figure_caption` | Caption starting with `Figure N.` or `Fig. N.` |
| `table_caption` | Caption starting with `Table N.` |
| `reference` | Individual numbered reference entry under a REFERENCES heading |
| `back_matter` | Content under ACKNOWLEDGMENTS, AUTHOR CONTRIBUTIONS, etc. |

## Deterministic Parsing Rules

The parser processes the normalized markdown **line by line** without any LLM call.

1. **Line classification** — A line is one of:
   - *Blank*: skipped, marks a paragraph boundary
   - *Heading*: starts with one or more `#` characters
   - *Content*: anything else

2. **Headings** update `section_path`:
   - Level 1 `#` → `section_path = [heading_text]`
   - Level 2 `##` → `section_path = [prev_L1, heading_text]`
   - Level 3 `###` → `section_path = [prev_L1, prev_L2, heading_text]`

3. **Paragraph merging** — consecutive content lines (no blank line between them)
   form a single paragraph block. The block text is the lines joined with a space,
   then stripped.

4. **References mode** is entered when a heading whose text matches
   `REFERENCES`, `REFERENCE LIST`, or `BIBLIOGRAPHY` (case-insensitive) is
   encountered. In references mode:
   - Lines matching `^\d+\.\s` start a new reference block
   - Continuation lines are appended to the previous reference block
   - References mode ends when a new heading is encountered

5. **Back-matter mode** is entered on headings matching:
   `ACKNOWLEDGMENTS`, `AUTHOR CONTRIBUTIONS`, `DECLARATION OF INTERESTS`,
   `SUPPLEMENTAL INFORMATION`, `DATA AND CODE AVAILABILITY`, `COMPETING INTERESTS`,
   `FUNDING` (all case-insensitive). Content under these headings gets
   `block_type = "back_matter"`.

6. **Figure captions** — a content group whose first line matches
   `^(Figure|Fig\.?)\s+\d+` (case-insensitive) is classified as `figure_caption`.

7. **Table captions** — a content group whose first line matches
   `^Table\s+\d+` (case-insensitive) is classified as `table_caption`.

8. **Title detection** — the first non-artifact heading encountered in the `# Text`
   section that (a) is at heading level 1, (b) has ≥ 20 characters, and (c) is not
   all-uppercase gets `block_type = "title"`.

9. **Step 2 artifact sections** (`# Images`, `# Captions`, `# Tables`, `# Text`)
   are treated as regular headings; their content follows the same rules.
   Image-link lines inside `# Images` are silently skipped.

10. **Text preservation** — the parser never rewrites, summarizes, spell-checks,
    or normalizes the text. Only leading/trailing whitespace is stripped per block.

## Why No LLM?

LLMs introduce non-determinism: the same input can produce different block
boundaries or types across runs. For a RAG pipeline:

- **Reproducibility**: the same normalized.md must always produce the same blocks
  so that block IDs are stable references.
- **Auditability**: every classification decision is traceable to a regex or line
  pattern — no black box.
- **Speed and cost**: parsing a full article takes milliseconds and zero API cost.
- **Correctness**: structure (headings, references, captions) is signaled by
  explicit markup patterns that are more reliably detected by regex than by an LLM
  operating on ambiguous prose.

LLMs are reserved for steps that require semantic understanding
(normalization in Step 2, embedding in Step 4+).

## Running Step 3A

```bash
# Single article
python pipeline/step3_structure_normalized.py --article-id pdf_16edbbde296287d6

# All articles
python pipeline/step3_structure_normalized.py --all

# Overwrite existing output
python pipeline/step3_structure_normalized.py --all --overwrite
```

## Auditing Output

```bash
# All indexed articles
python scripts/audit_structured_blocks.py

# Single article
python scripts/audit_structured_blocks.py --article-id pdf_16edbbde296287d6

# JSON output
python scripts/audit_structured_blocks.py --json
```

## Known Limitations

- **References without a REFERENCES heading**: articles that list numbered
  references without an explicit heading (e.g., inline at end of body text) will
  have those items classified as `paragraph` rather than `reference`.
- **Continued reference lines**: reference entries that span multiple lines are
  joined only if the continuation lines do not themselves start with `^\d+\.`.
- **Nested headings**: heading levels deeper than the actual article structure
  (e.g., `####`) are parsed correctly but may produce unusual `section_path` depths.
- **Token count**: `token_count` is a word-count approximation; it does not
  account for subword tokenization of scientific symbols and LaTeX fragments.
- **Step 2 artifacts in body text**: captions or image links that were NOT moved
  to `# Captions` / `# Images` by Step 2 may appear as `paragraph` blocks.
