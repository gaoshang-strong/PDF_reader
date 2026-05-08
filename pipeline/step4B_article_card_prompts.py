"""Prompt templates for Article Literature Card module extraction (Step 4B).

This module is LLM-free. It defines and provides prompt strings that Step 4C
will combine with article_reading_text and send to the LLM.
"""

from __future__ import annotations

_MODULES: list[str] = [
    "research_question",
    "study_design",
    "samples_and_cohorts",
    "data_source_and_provenance",
    "omics_and_experimental_methods",
    "computational_analysis_pipeline",
    "key_findings",
    "mechanism_model",
    "author_reported_limitations_and_future_directions",
]

_PLACEHOLDER = "{article_reading_text}"

_HEADER = (
    "You are an expert biomedical scientist extracting structured information "
    "from a scientific article for a literature database.\n\n"
)

# Used as the system message in the cached 2-message design (Step 4C).
# Keep wording consistent with _HEADER so full-prompt and cached modes are equivalent.
_SYSTEM_PRELUDE = (
    "You are an expert biomedical scientist extracting structured information "
    "from a scientific article for a literature database. "
    "Return only a valid JSON object. No markdown fences, no prose, no explanation."
)

_RULES = (
    "Rules:\n"
    "- Return JSON only. No markdown fences, no prose before or after the JSON object.\n"
    "- Return only this module object — not the full article card.\n"
    "- Match the schema exactly. Use only the fields shown in the JSON skeleton. "
    "Do not add extra fields.\n"
    "- Do not invent information. Use null or [] when information is not found in the text.\n"
    "- Evidence: use only parent_id values that appear verbatim in the article reading text. "
    "Do not fabricate parent_ids.\n"
    "- Do not critique evidence quality, study design, or methods.\n"
    "- Do not generate hypotheses or interpretations beyond what the paper explicitly states.\n"
    "- If the module contains any non-empty items, claims, comparison_groups, or meaningful "
    "extracted content, write a concise non-null summary (1-3 sentences).\n"
    "- Limit module-level evidence to at most 5 evidence objects.\n"
    "- For each item, claim, or comparison group, include at most 2 strongest evidence objects.\n"
    "- Avoid duplicate evidence from the same parent_id unless it supports different extracted facts.\n"
    "- Prefer evidence that directly states the extracted fact.\n"
    "- Use supporting_text as a short verbatim quote or close paraphrase from the article text.\n"
    "- Do not over-extract minor technical details unless they are important to understanding the study.\n"
    "- For omics_and_experimental_methods and computational_analysis_pipeline, prioritize major "
    "assays, major perturbations, major datasets, and major analysis steps."
)

_EVIDENCE_STRUCTURE = (
    "Evidence object structure (for all evidence arrays):\n"
    "{\n"
    '  "parent_id": "<verbatim parent_id from article reading text>",\n'
    '  "section_path": ["<string>", "..."],\n'
    '  "supporting_text": "<verbatim or near-verbatim passage, or null>",\n'
    '  "char_start": null,\n'
    '  "char_end": null\n'
    "}"
)


def _assemble(
    module_label: str, goal: str, questions: str, skeleton: str, notes: str = ""
) -> tuple[str, str]:
    """Return (full_prompt, module_question).

    full_prompt:     single-message prompt with article text embedded (backward compat).
    module_question: user message for the cached 2-message design (no article text).
    """
    core_parts = [
        "Goal:\n" + goal,
        "Expert reading questions:\n" + questions,
        _RULES,
        _EVIDENCE_STRUCTURE,
    ]
    if notes:
        core_parts.append("Schema notes:\n" + notes)
    core_parts.append(
        "JSON skeleton — return exactly this structure with all fields filled in:\n" + skeleton
    )
    module_question = (
        "Extract the [" + module_label + "] module.\n\n"
        + "\n\n".join(core_parts)
    )
    full_prompt = (
        _HEADER
        + module_question
        + "\n\nArticle reading text:\n" + _PLACEHOLDER
    )
    return full_prompt, module_question


# ---------------------------------------------------------------------------
# Module prompt templates
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {}
_QUESTIONS: dict[str, str] = {}

# A. research_question
_PROMPTS["research_question"], _QUESTIONS["research_question"] = _assemble(
    module_label="research_question",
    goal=(
        "Identify the biomedical problem the paper addresses, the central scientific question, "
        "any stated knowledge gap or unmet need, and the study type."
    ),
    questions=(
        "- What disease, condition, biological system, or biomedical problem does this paper study?\n"
        "- What is the central scientific question the authors set out to answer?\n"
        "- What gap in knowledge or unmet need do the authors explicitly state?\n"
        "- What type of study is this? Examples: mechanism, biomarker discovery, therapy response, "
        "data resource, method development, atlas, review."
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "disease_context": [],\n'
        '  "biological_context": [],\n'
        '  "main_question": null,\n'
        '  "stated_gap": null,\n'
        '  "study_type": [],\n'
        '  "evidence": []\n'
        "}"
    ),
)

# B. study_design
_PROMPTS["study_design"], _QUESTIONS["study_design"] = _assemble(
    module_label="study_design",
    goal=(
        "Describe how the study was structured: overall design, comparison groups, "
        "discovery/validation structure, and any intervention or perturbation context."
    ),
    questions=(
        "- What is the overall study design?\n"
        "- Is there a discovery/validation structure? If so, describe it.\n"
        "- What are the main comparison groups "
        "(e.g. tumor-normal, responder-nonresponder, pre-post treatment, high-low expression)?\n"
        "- Is there a longitudinal, intervention, perturbation, or model-based design?\n"
        "- What intervention, treatment, or experimental perturbation is described?"
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "design_features": [],\n'
        '  "comparison_groups": [\n'
        "    {\n"
        '      "group_a": null,\n'
        '      "group_b": null,\n'
        '      "comparison_purpose": null,\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "discovery_validation_structure": null,\n'
        '  "perturbation_or_intervention": null,\n'
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "Set comparison_groups to [] if no comparison groups are described. "
        "design_features must be a list of plain strings (e.g. ['retrospective', 'multi-cohort']), not a list of objects. "
        "discovery_validation_structure must be a plain string (e.g. 'Discovery cohort: N=50; Validation cohort: N=80'), not an object. "
        "perturbation_or_intervention must be a plain string describing the intervention(s), not an object."
    ),
)

# C. samples_and_cohorts
_PROMPTS["samples_and_cohorts"], _QUESTIONS["samples_and_cohorts"] = _assemble(
    module_label="samples_and_cohorts",
    goal=(
        "Enumerate all cohorts, sample groups, models, and biological materials used. "
        "Capture species, sample material, size, pairing, treatment context, timepoint, "
        "sorting strategy, and what each sample set was used for."
    ),
    questions=(
        "- What cohorts, samples, models, or biological materials are used?\n"
        "- Are they human, mouse, cell line, organoid, primary cells, or other systems?\n"
        "- What sample materials are used: tumor tissue, adjacent normal, blood, PBMC, "
        "TIL, sorted cells, biopsies, cell lines?\n"
        "- How many samples or patients are reported?\n"
        "- Are samples paired (e.g. tumor–normal from the same patient)?\n"
        "- Are treatment context, timepoint, sorting, or enrichment strategy described?\n"
        "- What is each cohort or sample set used for (discovery, validation, functional assay, etc.)?"
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "items": [\n'
        "    {\n"
        '      "sample_or_cohort_name": null,\n'
        '      "species": null,\n'
        '      "disease_or_condition": null,\n'
        '      "sample_material": [],\n'
        '      "sample_count": null,\n'
        '      "paired_design": null,\n'
        '      "treatment_context": null,\n'
        '      "timepoint": null,\n'
        '      "sorting_or_enrichment": null,\n'
        '      "used_for": [],\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "items: one object per distinct cohort or sample group. Set items to [] if none described.\n"
        "sample_count: integer number of samples or patients, or null if not reported.\n"
        "paired_design: true if samples are explicitly paired (e.g. tumor–normal from same patient), "
        "false if unpaired, null if not stated."
    ),
)

# D. data_source_and_provenance
_PROMPTS["data_source_and_provenance"], _QUESTIONS["data_source_and_provenance"] = _assemble(
    module_label="data_source_and_provenance",
    goal=(
        "Enumerate all datasets used, distinguishing data generated in this study from "
        "data obtained from external sources, and capturing accession IDs and usage."
    ),
    questions=(
        "- Which data were generated in this study?\n"
        "- Which data came from public databases, external cohorts, or previously published studies?\n"
        "- Are TCGA, GEO, ArrayExpress, SRA, dbGaP, CPTAC, or other repositories mentioned?\n"
        "- Are accession IDs, dataset names, or deposit references reported?\n"
        "- What was each dataset used for in this study?"
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "items": [\n'
        "    {\n"
        '      "dataset_name": null,\n'
        '      "data_modality": null,\n'
        '      "source_type": "generated_in_this_study | external_public_data | external_published_cohort | external_database | unclear",\n'
        '      "external_database_or_cohort": null,\n'
        '      "accession_id": null,\n'
        '      "generated_in_this_study": null,\n'
        '      "used_for": null,\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "source_type must be exactly one of: generated_in_this_study, external_public_data, "
        "external_published_cohort, external_database, unclear.\n"
        "accession_id: the repository accession string (e.g. GSE123456, phs000178), or null.\n"
        "Set items to [] if no datasets are described."
    ),
)

# E. omics_and_experimental_methods
_PROMPTS["omics_and_experimental_methods"], _QUESTIONS["omics_and_experimental_methods"] = _assemble(
    module_label="omics_and_experimental_methods",
    goal=(
        "Enumerate all assays, omics methods, and wet-lab methods used. "
        "For each, capture the data type it produces, sample material used, "
        "whether it was generated in this study, and its purpose."
    ),
    questions=(
        "- What assays, omics methods, and wet-lab methods are used?\n"
        "- Look for: scRNA-seq, snRNA-seq, bulk RNA-seq, ATAC-seq, scATAC-seq, "
        "spatial transcriptomics, proteomics, phosphoproteomics, metabolomics, "
        "TCR-seq, BCR-seq, CITE-seq, CyTOF, flow cytometry, IHC, IF, immunofluorescence, "
        "CRISPR, perturbation, co-culture, organoid, mouse model, and validation experiments.\n"
        "- What data type does each method produce?\n"
        "- What sample material does each method use?\n"
        "- Was the data generated in this study, or is it from external sources?\n"
        "- What was the purpose of each method?"
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "items": [\n'
        "    {\n"
        '      "method": null,\n'
        '      "data_type": null,\n'
        '      "modality_category": null,\n'
        '      "sample_material": null,\n'
        '      "generated_in_this_study": null,\n'
        '      "purpose": null,\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "modality_category: use one of the following recommended values where applicable: "
        "transcriptomics, single_cell_transcriptomics, spatial_transcriptomics, proteomics, "
        "metabolomics, epigenomics, immune_profiling, flow_cytometry, imaging, "
        "functional_assay, perturbation, in_vivo_model, computational_reanalysis, "
        "clinical_data, other.\n"
        "generated_in_this_study: true if produced by the authors in this study, false if from "
        "an external source, null if unclear.\n"
        "Set items to [] if no methods are described."
    ),
)

# F. computational_analysis_pipeline
_PROMPTS["computational_analysis_pipeline"], _QUESTIONS["computational_analysis_pipeline"] = _assemble(
    module_label="computational_analysis_pipeline",
    goal=(
        "Enumerate the computational analysis steps used. For each step, capture the "
        "software or method, input data, output, and any reported parameters or thresholds."
    ),
    questions=(
        "- What computational analyses are described?\n"
        "- Look for: QC, alignment, preprocessing, normalization, batch correction, "
        "integration, clustering, cell type annotation, differential expression, "
        "pathway analysis, gene set enrichment, trajectory inference, pseudotime, "
        "RNA velocity, CNV inference, ligand-receptor analysis, cell-cell communication, "
        "spatial analysis, survival analysis, multi-omics integration, statistical models, "
        "and machine learning.\n"
        "- What software, package, database, or method is used for each step?\n"
        "- What input data and output results are described?\n"
        "- Are parameters, thresholds, cutoffs, or versions reported?"
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "items": [\n'
        "    {\n"
        '      "analysis_step": null,\n'
        '      "software_or_method": [],\n'
        '      "input_data": null,\n'
        '      "output": null,\n'
        '      "parameters_or_thresholds": null,\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "software_or_method: list of software names and versions used for this step, e.g. "
        '["Seurat 4.0", "Harmony"].\n'
        "parameters_or_thresholds: a concise string summarising key parameters, e.g. "
        '"resolution=0.5, min.cells=3, FDR<0.05".\n'
        "Set items to [] if no computational steps are described."
    ),
)

# G. key_findings
_PROMPTS["key_findings"], _QUESTIONS["key_findings"] = _assemble(
    module_label="key_findings",
    goal=(
        "Extract the main findings and conclusions reported by the paper. "
        "For each finding, capture the supporting data, entities involved, and result type."
    ),
    questions=(
        "- What are the main findings or conclusions the authors report?\n"
        "- What data, assay, or analysis supports each finding?\n"
        "- What cell types, genes, proteins, pathways, biomarkers, targets, phenotypes, "
        "clinical associations, or therapy-response patterns are involved?\n"
        "- What kind of result is each finding: association, causal evidence, classification, "
        "validation of prior work, atlas or resource, mechanism-related observation, "
        "therapeutic implication, or clinical finding?"
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "items": [\n'
        "    {\n"
        '      "finding": null,\n'
        '      "associated_data": [],\n'
        '      "associated_entities": [],\n'
        '      "result_type": null,\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "finding: a single declarative sentence stating the finding as reported by the authors.\n"
        "associated_data: assays or datasets that support this finding, e.g. "
        '["scRNA-seq", "IHC", "TCGA validation"].\n'
        "associated_entities: specific biological entities, e.g. "
        '["CD8+ T cells", "CXCL12", "PNI score"].\n'
        "result_type: one of: association, causal_evidence, classification, "
        "validation, atlas_resource, therapeutic_implication, clinical_finding, or other.\n"
        "Set items to [] if no findings can be extracted."
    ),
)

# H. mechanism_model
_PROMPTS["mechanism_model"], _QUESTIONS["mechanism_model"] = _assemble(
    module_label="mechanism_model",
    goal=(
        "Extract any mechanistic model or biological mechanism proposed by the authors. "
        "If no mechanism is proposed, set summary to null and claims to []."
    ),
    questions=(
        "- Does the paper propose a mechanism or biological model?\n"
        "- If yes: what is the mechanism chain (ordered sequence of events)?\n"
        "- What is the upstream factor that initiates the mechanism?\n"
        "- What mediators (intermediate molecules, cells, signals) are involved?\n"
        "- What downstream effect or phenotype is proposed?\n"
        "- What cell types, pathways, molecules, ligand-receptor pairs, cytokines, "
        "metabolites, or transcription factors are involved?\n"
        "- What therapeutic or clinical implication does the paper describe?\n"
        "- If no mechanism is proposed, set summary to null and claims to []."
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "claims": [\n'
        "    {\n"
        '      "mechanism_chain": [],\n'
        '      "upstream_factor": null,\n'
        '      "mediators": [],\n'
        '      "downstream_effect": null,\n'
        '      "involved_cell_types": [],\n'
        '      "involved_pathways_or_molecules": [],\n'
        '      "therapeutic_or_clinical_implication": null,\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "mechanism_chain: ordered list of steps, e.g. "
        '["PNI", "Schwann cell activation", "CXCL12 secretion", "T cell exclusion"].\n'
        "Do not infer a mechanism if the paper only reports an association without proposing causality.\n"
        "Set claims to [] if no mechanism is proposed."
    ),
)

# I. author_reported_limitations_and_future_directions
_PROMPTS["author_reported_limitations_and_future_directions"], _QUESTIONS["author_reported_limitations_and_future_directions"] = _assemble(
    module_label="author_reported_limitations_and_future_directions",
    goal=(
        "Extract only limitations, caveats, and future directions that the authors explicitly state. "
        "Do not add your own critique."
    ),
    questions=(
        "- What limitations do the authors explicitly report?\n"
        "- What caveats do the authors explicitly mention?\n"
        "- What future directions do the authors explicitly propose?\n"
        "- Only extract what the authors themselves say. "
        "Do not add your own assessment of the study's weaknesses."
    ),
    skeleton=(
        "{\n"
        '  "summary": null,\n'
        '  "items": [\n'
        "    {\n"
        '      "type": "limitation | caveat | future_direction",\n'
        '      "text": null,\n'
        '      "evidence": []\n'
        "    }\n"
        "  ],\n"
        '  "evidence": []\n'
        "}"
    ),
    notes=(
        "type must be exactly one of: limitation, caveat, future_direction.\n"
        "limitation: a constraint the authors acknowledge (e.g. small sample size, single-centre).\n"
        "caveat: a qualifier on interpretation (e.g. correlation does not prove causation).\n"
        "future_direction: a follow-up study or experiment the authors propose.\n"
        "Set items to [] if none are explicitly stated."
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_article_card_modules() -> list[str]:
    """Return the ordered list of Article Literature Card module names."""
    return list(_MODULES)


def get_article_card_module_prompt(module_name: str, article_reading_text: str) -> str:
    """Return the filled-in single-message prompt (article text embedded).

    Raises ValueError for unknown module names.
    """
    if module_name not in _PROMPTS:
        raise ValueError(
            f"Unknown module name: {module_name!r}. "
            f"Valid modules: {_MODULES}"
        )
    return _PROMPTS[module_name].replace(_PLACEHOLDER, article_reading_text)


def get_article_card_module_question(module_name: str) -> str:
    """Return the module extraction question (user message for cached extraction).

    Does not contain the article reading text — pair with
    get_article_reading_system_prompt() as the system message.

    Raises ValueError for unknown module names.
    """
    if module_name not in _QUESTIONS:
        raise ValueError(
            f"Unknown module name: {module_name!r}. "
            f"Valid modules: {_MODULES}"
        )
    return _QUESTIONS[module_name]


def get_article_reading_system_prompt(article_reading_text: str) -> str:
    """Build the system message for cached extraction.

    Contains the extraction instructions and the full article text.
    This is the cacheable prefix — identical across all 9 module calls
    for the same article, enabling server-side prompt caching.
    """
    return (
        _SYSTEM_PRELUDE
        + "\n\nArticle reading text:\n"
        + article_reading_text
    )
