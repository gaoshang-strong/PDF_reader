# Article Literature Card Schema (v0.1.0)

## Purpose

An **Article Literature Card** is a structured JSON document that captures
objective factual content extracted from a single biomedical paper.  It is
the primary output of the LLM extraction step (Step 4B) and the primary input
to downstream retrieval, comparison, and synthesis.

One card corresponds to exactly one article.  The card does not answer
questions.  It does not score evidence quality.  It does not generate
hypotheses.  It records what the paper says, in a form that downstream code
can query and compare.

Schema file: `schemas/article_card.schema.json` (JSON Schema Draft 7)

---

## Top-Level Structure

```json
{
  "schema_version": "0.1.0",
  "article_id": "pdf_16edbbde296287d6",
  "metadata": { ... },
  "modules": { ... },
  "extraction_metadata": { ... }
}
```

### Required top-level fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Schema version; must be present for forward-compatibility checks |
| `article_id` | string | Matches the article folder name under `data/articles/` |
| `metadata` | object | Bibliographic fields |
| `modules` | object | Nine extraction modules (A–I) |
| `extraction_metadata` | object | Provenance of the extraction run |

### `metadata`

| Field | Type |
|---|---|
| `title` | string \| null |
| `authors` | string[] |
| `journal` | string \| null |
| `year` | integer \| null |
| `doi` | string \| null |
| `article_type` | string \| null |

### `extraction_metadata`

| Field | Type | Notes |
|---|---|---|
| `extraction_method` | string \| null | e.g. `"llm:deepseek-v4-flash"` |
| `created_at` | string \| null | ISO-8601 timestamp |
| `source_files` | string[] | Paths to input files used during extraction |
| `module_status` | object | Free-form key→value map; used to record per-module extraction status |
| `warnings` | string[] | Non-fatal issues noted during extraction |

---

## Modules (A–I)

All nine modules are required.  Every module has at least a `summary` field
(string or null) and an `evidence` array.  Missing information is represented
as `null` or `[]` — never omitted.

### A. `research_question`

What problem does the paper address?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `disease_context` | string[] |
| `biological_context` | string[] |
| `main_question` | string \| null |
| `stated_gap` | string \| null |
| `study_type` | string[] |
| `evidence` | evidence_object[] |

### B. `study_design`

How was the study structured?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `design_features` | string[] |
| `comparison_groups` | comparison_group_object[] |
| `discovery_validation_structure` | string \| null |
| `perturbation_or_intervention` | string \| null |
| `evidence` | evidence_object[] |

### C. `samples_and_cohorts`

What biological material was studied?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `items` | sample_object[] |
| `evidence` | evidence_object[] |

### D. `data_source_and_provenance`

Where did the data come from?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `items` | data_source_object[] |
| `evidence` | evidence_object[] |

### E. `omics_and_experimental_methods`

What assays and measurement technologies were used?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `items` | method_object[] |
| `evidence` | evidence_object[] |

### F. `computational_analysis_pipeline`

What software and analytical steps were applied?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `items` | analysis_step_object[] |
| `evidence` | evidence_object[] |

### G. `key_findings`

What did the paper discover or report?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `items` | finding_object[] |
| `evidence` | evidence_object[] |

### H. `mechanism_model`

What mechanistic claims does the paper make?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `claims` | mechanism_claim_object[] |
| `evidence` | evidence_object[] |

### I. `author_reported_limitations_and_future_directions`

What does the paper itself acknowledge as limitations or open questions?

| Field | Type |
|---|---|
| `summary` | string \| null |
| `items` | limitation_object[] |
| `evidence` | evidence_object[] |

---

## Common Object Definitions

### `evidence_object`

Links any extracted fact back to a parent chunk in the index.

```json
{
  "parent_id": "pdf_abc::parent_000003",
  "section_path": ["Text", "Results", "Survival analysis"],
  "supporting_text": "CD8+ T cells were significantly reduced in PNI+ tumors.",
  "char_start": 14210,
  "char_end": 14280
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `parent_id` | string | Yes | Matches a `chunk_id` in `parents.jsonl` |
| `section_path` | string[] | Yes | Section hierarchy from the parent chunk |
| `supporting_text` | string \| null | No | Verbatim or near-verbatim text from the source |
| `char_start` | integer \| null | No | Byte offset into `normalized.md` |
| `char_end` | integer \| null | No | Byte offset into `normalized.md` |

### `comparison_group_object`

```json
{
  "group_a": "PNI-positive tumors",
  "group_b": "PNI-negative tumors",
  "comparison_purpose": "Survival and immune infiltration",
  "evidence": []
}
```

### `sample_object`

```json
{
  "sample_or_cohort_name": "Discovery cohort",
  "species": "Human",
  "disease_or_condition": "Colorectal cancer",
  "sample_material": ["tumor tissue", "adjacent normal"],
  "sample_count": 82,
  "paired_design": true,
  "treatment_context": "Treatment-naive",
  "timepoint": "Pre-treatment",
  "sorting_or_enrichment": null,
  "used_for": ["scRNA-seq", "IHC"],
  "evidence": []
}
```

### `data_source_object`

`source_type` is a controlled enum:

| Value | Meaning |
|---|---|
| `generated_in_this_study` | Data produced by the authors in this study |
| `external_public_data` | Data downloaded from a public repository |
| `external_published_cohort` | Data from a previously published cohort |
| `external_database` | Reference database (e.g. MSigDB, UniProt) |
| `unclear` | Provenance not determinable from the text |

```json
{
  "dataset_name": "TCGA-CRC",
  "data_modality": "bulk RNA-seq",
  "source_type": "external_public_data",
  "external_database_or_cohort": "TCGA",
  "accession_id": "phs000178",
  "generated_in_this_study": false,
  "used_for": "Validation of survival associations",
  "evidence": []
}
```

### `method_object`

```json
{
  "method": "10x Genomics scRNA-seq",
  "data_type": "single-cell transcriptomics",
  "modality_category": "transcriptomics",
  "sample_material": "tumor tissue",
  "generated_in_this_study": true,
  "purpose": "Characterize immune cell composition",
  "evidence": []
}
```

### `analysis_step_object`

```json
{
  "analysis_step": "Cell clustering",
  "software_or_method": ["Seurat 4.0"],
  "input_data": "Filtered scRNA-seq count matrix",
  "output": "UMAP clusters with cell-type annotations",
  "parameters_or_thresholds": "resolution=0.5, dims=1:30",
  "evidence": []
}
```

### `finding_object`

```json
{
  "finding": "PNI correlates with reduced CD8+ T cell infiltration.",
  "associated_data": ["scRNA-seq", "IHC"],
  "associated_entities": ["CD8+ T cells", "PNI score"],
  "result_type": "association",
  "evidence": []
}
```

### `mechanism_claim_object`

```json
{
  "mechanism_chain": ["PNI", "Schwann cell activation", "CXCL12 secretion", "T cell exclusion"],
  "upstream_factor": "Perineural invasion",
  "mediators": ["Schwann cells", "CXCL12"],
  "downstream_effect": "CD8+ T cell exclusion from tumor core",
  "involved_cell_types": ["Schwann cells", "CD8+ T cells"],
  "involved_pathways_or_molecules": ["CXCL12–CXCR4 axis"],
  "therapeutic_or_clinical_implication": "CXCR4 blockade may restore T cell infiltration",
  "evidence": []
}
```

### `limitation_object`

`type` is a controlled enum: `limitation`, `caveat`, `future_direction`.

```json
{
  "type": "limitation",
  "text": "The study is limited to colorectal cancer and may not generalize to other tumor types.",
  "evidence": []
}
```

---

## Validation

```bash
# Validate one file
python scripts/validate_article_card.py data/cards/pdf_abc.json

# Validate multiple files
python scripts/validate_article_card.py data/cards/*.json
```

Exit code 0 = all files valid.  Exit code 1 = one or more failures.

---

## What Is Intentionally Excluded

The following are **not** part of this schema, by design:

| Excluded concept | Reason |
|---|---|
| **Topic relevance scoring** | Whether a paper is relevant to a user query is a retrieval concern, not an extraction concern. |
| **Hypothesis generation** | The card records what the paper says; generating new hypotheses from it is a separate downstream step. |
| **Evidence strength critique** | Judging study quality (risk of bias, effect size reliability) is a reviewer function, not objective extraction. The card records author-reported limitations only (module I). |
| **Cross-article relationships** | Linking findings across papers (e.g. confirmations, contradictions) belongs to a synthesis layer, not the card. |
| **Free-text LLM commentary** | All fields are typed and schema-validated; no open-ended prose fields exist at the top level. |
