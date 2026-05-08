# PDF_reader

## End-to-end purpose

Transforms raw PDF parse outputs (MinerU markdown + GROBID metadata) into structured, RAG-ready representations. The pipeline produces token-bounded parent and child chunks for retrieval, plus a nine-module literature card (JSON + Markdown) summarising each article.

## Required input structure

Each article needs a MinerU markdown file. A GROBID TEI XML file is optional but required for title, authors, DOI, and year.

```
data/
  parser_outputs/
    mineru_work/
      {article_id}/
        {article_id}/
          full.md
          images/          # optional
    grobid_tei/
      {article_id}.tei.xml # optional; metadata will be empty without it
```

`article_id` is the directory name used consistently across all pipeline outputs (e.g. `pdf_16edbbde296287d6`).

Steps 2 and 4 also require an LLM API key:

```
DEEPSEEK_API_KEY=...        # or OPENAI_API_KEY for compatible endpoints
DEEPSEEK_BASE_URL=...       # optional; defaults to https://api.deepseek.com
```

## How to run

Run the steps in order. Each step is independent and reads from the previous step's output.

```bash
# Step 1 — canonicalise articles (copy MinerU output, extract GROBID metadata)
python pipeline/step1_canonical_articles.py [--article-id ID] [--overwrite]

# Step 2 — normalise markdown with LLM
python pipeline/step2_normalize.py [--article-id ID] [--overwrite] [--model MODEL]

# Step 3 — deterministic structuring: blocks → parents → children
python pipeline/step3_structure_normalized.py [--article-id ID] [--all] [--overwrite]
python pipeline/step3_build_parents.py        [--article-id ID] [--all] [--overwrite]
python pipeline/step3_build_children.py       [--article-id ID] [--all] [--overwrite]

# Step 4 — extract and render literature cards (batch runner for steps 4C + 4D)
python pipeline/step4_batch_run.py [--article-ids ID1 ID2 ...] [--overwrite] [--skip-render]
```

`--overwrite` re-runs and overwrites existing output; without it, already-processed articles are skipped.  
`--skip-render` (step 4 only) extracts `article_card.json` but skips rendering `article_card.md`.

## Output structure

```
data/
  articles/
    {article_id}/
      article_text.raw.md          # copied from MinerU
      article_text.normalized.md   # LLM-normalised (step 2)
      metadata.json                # title, authors, DOI, year, journal
      images/
  index/
    {article_id}/
      structured_blocks.jsonl      # deterministic block segmentation
      parents.jsonl                # parent chunks (1 200–1 800 tokens)
      parent_manifest.json
      children.jsonl               # child chunks (380–550 tokens, with overlap)
      child_manifest.json
  literature_cards/
    {article_id}/
      article_card.json            # structured 9-module extraction
      article_card.md              # human-readable rendering
      extraction_manifest.json
      modules/                     # per-module raw LLM outputs
```

## How to interpret the output

**`article_text.normalized.md`** — LLM-restructured version of the raw markdown. Reorganised into four sections (Text, Images, Captions, Tables) with repaired sentences and normalised LaTeX.

**`structured_blocks.jsonl`** — One JSON object per line. Each block has a `block_type` (`heading`, `paragraph`, `figure_caption`, `table_caption`, `reference`, `back_matter`), a `section_path` array, and `char_start`/`char_end` offsets into `normalized.md`.

**`parents.jsonl`** — Token-bounded parent chunks for RAG retrieval. Key fields: `parent_id`, `text`, `section_path`, `chunk_type`, `token_count`, `include_in_default_qa` (false for back-matter). Format: `{article_id}::parent_{index:06d}`.

**`children.jsonl`** — Smaller splits of each parent, with overlap. Each child has `parent_id`, `text_for_embedding`, and `text_for_bm25` variants for dense and sparse retrieval. Format: `{article_id}::child_{index:06d}`.

**`article_card.json`** — Structured literature card with top-level keys: `metadata`, `modules`, `extraction_metadata`. The `modules` dict contains nine entries (e.g. `research_question`, `key_findings`, `study_design`). Each module has a `summary`, an `evidence` array, and module-specific structured fields. Each evidence object cites a `parent_id` and includes the supporting text span.

**`article_card.md`** — Human-readable rendering of `article_card.json`. Sections A–I correspond to the nine modules. Each section ends with an **Evidence parents** line listing the `parent_id` values that grounded the extraction.

**`extraction_manifest.json`** — Records extraction timestamp, model, per-module status, and any warnings (e.g. missing metadata, module parse failures).
