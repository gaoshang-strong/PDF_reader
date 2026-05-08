# Step 2: Normalize Article Markdown

## Goal

Transform the raw MinerU Markdown (`article_text.raw.md`) into a clean,
consistently structured document (`article_text.normalized.md`). The
normalized file is what every downstream step reads — it is the single
source of truth for article text.

---

## Key Idea

MinerU output is noisy: inconsistent heading levels, mid-sentence line
breaks, garbled LaTeX, mixed image/table/text interleaving. A one-shot LLM
call (DeepSeek via OpenAI-compatible API) reorganizes the text into four
canonical top-level sections:

```
# Text          — all prose content, headings preserved
# Images        — image links only
# Captions      — figure and table captions
# Tables        — table content
```

The LLM is instructed to **reorganize, not summarize** — total text length
must be preserved. After normalization, a validation step checks the
length ratio (normalized ÷ raw); a ratio outside 0.95–1.05 is flagged as
a warning.

**LaTeX handling:** text-like math (e.g. `$\mathtt{CD8}^{+}$` → `CD8+`)
is converted to plain text. Genuine equations are preserved.

**FAIL articles:** articles flagged as FAIL in Step 1 (> 115k tokens) skip
the LLM call entirely. Their raw Markdown is copied to
`article_text.normalized.md` verbatim.

---

## Inputs

| Source | Path |
|---|---|
| Raw Markdown | `data/articles/{article_id}/article_text.raw.md` |
| Length report | `data/articles/{article_id}/raw_article_report.json` |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | Primary API key |
| `OPENAI_API_KEY` | — | Fallback API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | LLM endpoint |

---

## Outputs

| File | Path | Description |
|---|---|---|
| Normalized Markdown | `data/articles/{article_id}/article_text.normalized.md` | Clean, structured Markdown with four top-level sections |

---

## How to Run

```bash
export DEEPSEEK_API_KEY=sk-...

# All articles
python pipeline/step2_normalize.py --all

# Single article
python pipeline/step2_normalize.py --article-id pdf_b984e5bc1768479a

# Re-normalize already-processed articles
python pipeline/step2_normalize.py --all --overwrite

# Skip articles that already have a normalized file
python pipeline/step2_normalize.py --all --skip-existing

# Custom model
python pipeline/step2_normalize.py --all --model deepseek-chat
```

---

## Python API

```python
from pathlib import Path
from pipeline.step2_normalize import make_client, process_article

client = make_client()  # reads DEEPSEEK_API_KEY from env

result = process_article(
    "pdf_b984e5bc1768479a",
    data_root=Path("data"),
    client=client,
    overwrite=True,
)
# result = {"article_id": ..., "status": "PASS", "ratio": 1.01, "missing_headings": []}
```

---

## Validation

After writing the normalized file, the step validates:

| Check | Pass condition |
|---|---|
| Required headings | `# Text`, `# Images`, `# Captions`, `# Tables` all present |
| Length ratio | 0.95 ≤ (normalized chars) / (raw chars) ≤ 1.05 |

A failed validation adds a `WARN` status to the result but does **not**
abort — the normalized file is still written and used by downstream steps.
