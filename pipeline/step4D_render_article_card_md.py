#!/usr/bin/env python3
"""Step 4D: Render an Article Literature Card JSON as human-readable Markdown.

Input:  data/literature_cards/{article_id}/article_card.json
Output: data/literature_cards/{article_id}/article_card.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step4B_article_card_prompts import list_article_card_modules

ALL_MODULES = list_article_card_modules()

_MODULE_LABELS = {
    "research_question":                              "A. Research Question",
    "study_design":                                   "B. Study Design",
    "samples_and_cohorts":                            "C. Samples & Cohorts",
    "data_source_and_provenance":                     "D. Data Sources & Provenance",
    "omics_and_experimental_methods":                 "E. Omics & Experimental Methods",
    "computational_analysis_pipeline":                "F. Computational Pipeline",
    "key_findings":                                   "G. Key Findings",
    "mechanism_model":                                "H. Mechanism Model",
    "author_reported_limitations_and_future_directions": "I. Limitations & Future Directions",
}


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def collect_evidence_parent_ids(obj: object, seen: set[str] | None = None) -> list[str]:
    """Recursively collect unique parent_ids from all evidence arrays."""
    if seen is None:
        seen = set()
    result: list[str] = []
    if isinstance(obj, dict):
        for ev in obj.get("evidence") or []:
            pid = ev.get("parent_id", "")
            if pid and pid not in seen:
                seen.add(pid)
                result.append(pid)
        for v in obj.values():
            result.extend(collect_evidence_parent_ids(v, seen))
    elif isinstance(obj, list):
        for item in obj:
            result.extend(collect_evidence_parent_ids(item, seen))
    return result


def _evidence_line(module_data: dict) -> str:
    pids = collect_evidence_parent_ids(module_data)
    if not pids:
        return "_Evidence parents: none_"
    short = [pid.split("::")[-1] if "::" in pid else pid for pid in pids]
    return "**Evidence parents:** " + ", ".join(short)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _null_dash(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "—"
    return str(value)


def _summary_block(summary: str | None) -> str:
    return summary if summary else "_No summary extracted._"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, sep_row] + data_rows)


# ---------------------------------------------------------------------------
# Per-module renderers
# ---------------------------------------------------------------------------

def _render_research_question(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    dc = data.get("disease_context") or []
    bc = data.get("biological_context") or []
    mq = data.get("main_question")
    sg = data.get("stated_gap")
    st = data.get("study_type") or []
    if dc:
        lines.append(f"**Disease context:** {', '.join(dc)}")
    if bc:
        lines.append(f"**Biological context:** {', '.join(bc)}")
    if mq:
        lines.append(f"**Main question:** {mq}")
    if sg:
        lines.append(f"**Stated gap:** {sg}")
    if st:
        lines.append(f"**Study type:** {', '.join(st)}")
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_study_design(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    df = data.get("design_features") or []
    dvs = data.get("discovery_validation_structure")
    poi = data.get("perturbation_or_intervention")
    cg = data.get("comparison_groups") or []
    if df:
        lines.append("**Design features:** " + ", ".join(df))
    if dvs:
        lines.append(f"**Discovery/validation:** {dvs}")
    if poi:
        lines.append(f"**Intervention/perturbation:** {poi}")
    if cg:
        rows = [
            [_null_dash(g.get("group_a")), _null_dash(g.get("group_b")),
             _null_dash(g.get("comparison_purpose"))]
            for g in cg
        ]
        lines.append("")
        lines.append("**Comparison groups:**")
        lines.append(_md_table(["Group A", "Group B", "Purpose"], rows))
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_samples_and_cohorts(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    items = data.get("items") or []
    if items:
        rows = []
        for s in items:
            rows.append([
                _null_dash(s.get("sample_or_cohort_name")),
                _null_dash(s.get("species")),
                _null_dash(s.get("disease_or_condition")),
                str(s.get("sample_count")) if s.get("sample_count") is not None else "—",
                _null_dash(s.get("sample_material")),
                _null_dash(s.get("used_for")),
            ])
        lines.append("")
        lines.append(_md_table(["Cohort/Sample", "Species", "Disease", "N", "Material", "Used for"], rows))
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_data_source_and_provenance(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    items = data.get("items") or []
    if items:
        rows = []
        for d in items:
            rows.append([
                _null_dash(d.get("dataset_name")),
                _null_dash(d.get("data_modality")),
                _null_dash(d.get("source_type")),
                _null_dash(d.get("accession_id")),
            ])
        lines.append("")
        lines.append(_md_table(["Dataset", "Modality", "Source type", "Accession"], rows))
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_omics_and_experimental_methods(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    items = data.get("items") or []
    if items:
        rows = []
        for m in items:
            rows.append([
                _null_dash(m.get("method")),
                _null_dash(m.get("modality_category")),
                _null_dash(m.get("sample_material")),
                _null_dash(m.get("purpose")),
            ])
        lines.append("")
        lines.append(_md_table(["Method", "Category", "Material", "Purpose"], rows))
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_computational_analysis_pipeline(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    items = data.get("items") or []
    if items:
        rows = []
        for s in items:
            sw = s.get("software_or_method") or []
            rows.append([
                _null_dash(s.get("analysis_step")),
                ", ".join(sw) if sw else "—",
                _null_dash(s.get("parameters_or_thresholds")),
            ])
        lines.append("")
        lines.append(_md_table(["Step", "Software / Method", "Parameters"], rows))
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_key_findings(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    items = data.get("items") or []
    if items:
        lines.append("")
        for f in items:
            finding = f.get("finding") or "—"
            rt = f.get("result_type")
            tag = f" _{rt}_" if rt else ""
            lines.append(f"- {finding}{tag}")
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_mechanism_model(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    claims = data.get("claims") or []
    for i, c in enumerate(claims, 1):
        lines.append(f"\n**Claim {i}:**")
        chain = c.get("mechanism_chain") or []
        if chain:
            lines.append("- Chain: " + " → ".join(chain))
        uf = c.get("upstream_factor")
        meds = c.get("mediators") or []
        de = c.get("downstream_effect")
        cts = c.get("involved_cell_types") or []
        pms = c.get("involved_pathways_or_molecules") or []
        impl = c.get("therapeutic_or_clinical_implication")
        if uf:
            lines.append(f"- Upstream: {uf}")
        if meds:
            lines.append(f"- Mediators: {', '.join(meds)}")
        if de:
            lines.append(f"- Downstream: {de}")
        if cts:
            lines.append(f"- Cell types: {', '.join(cts)}")
        if pms:
            lines.append(f"- Pathways/molecules: {', '.join(pms)}")
        if impl:
            lines.append(f"- Implication: {impl}")
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


def _render_limitations(data: dict) -> str:
    lines = [_summary_block(data.get("summary"))]
    items = data.get("items") or []
    if items:
        lines.append("")
        type_icons = {"limitation": "⚠", "caveat": "~", "future_direction": "→"}
        for item in items:
            t = item.get("type", "limitation")
            icon = type_icons.get(t, "-")
            text = item.get("text") or "—"
            lines.append(f"- [{t}] {text}")
    lines.append("")
    lines.append(_evidence_line(data))
    return "\n".join(lines)


_MODULE_RENDERERS = {
    "research_question": _render_research_question,
    "study_design": _render_study_design,
    "samples_and_cohorts": _render_samples_and_cohorts,
    "data_source_and_provenance": _render_data_source_and_provenance,
    "omics_and_experimental_methods": _render_omics_and_experimental_methods,
    "computational_analysis_pipeline": _render_computational_analysis_pipeline,
    "key_findings": _render_key_findings,
    "mechanism_model": _render_mechanism_model,
    "author_reported_limitations_and_future_directions": _render_limitations,
}


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------

def render_article_card(card: dict) -> str:
    """Render article_card dict to a Markdown string."""
    parts: list[str] = []

    # --- Header ---
    article_id = card.get("article_id", "unknown")
    parts.append(f"# {article_id}")

    meta = card.get("metadata") or {}
    title = meta.get("title") or "_(title not extracted)_"
    parts.append(f"**Title:** {title}")

    authors = meta.get("authors") or []
    if authors:
        parts.append(f"**Authors:** {', '.join(authors)}")

    journal = meta.get("journal")
    year = meta.get("year")
    if journal or year:
        jy = " · ".join(str(x) for x in [journal, year] if x is not None)
        parts.append(f"**Journal:** {jy}")

    doi = meta.get("doi")
    if doi:
        parts.append(f"**DOI:** {doi}")

    em = card.get("extraction_metadata") or {}
    parts.append(f"**Extracted:** {em.get('created_at', '—')}  "
                 f"model: {em.get('extraction_method', '—')}")

    parts.append("---")

    # --- Modules ---
    modules = card.get("modules") or {}
    for module_name in ALL_MODULES:
        label = _MODULE_LABELS[module_name]
        data = modules.get(module_name) or {}
        renderer = _MODULE_RENDERERS[module_name]
        body = renderer(data)
        parts.append(f"## {label}\n\n{body}")
        parts.append("---")

    # --- Warnings ---
    warnings = em.get("warnings") or []
    if warnings:
        parts.append("## Extraction Warnings")
        for w in warnings:
            parts.append(f"- {w}")
        parts.append("---")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# File-level function
# ---------------------------------------------------------------------------

def render_article_card_file(
    article_id: str,
    *,
    overwrite: bool = False,
    data_root: Path | None = None,
) -> Path:
    """Read article_card.json and write article_card.md. Returns the output path."""
    root = data_root or (REPO_ROOT / "data")
    card_dir = root / "literature_cards" / article_id
    card_path = card_dir / "article_card.json"
    md_path = card_dir / "article_card.md"

    if not card_path.exists():
        raise FileNotFoundError(f"article_card.json not found: {card_path}")
    if md_path.exists() and not overwrite:
        raise FileExistsError(
            f"article_card.md already exists for '{article_id}'. Use --overwrite to replace."
        )

    with open(card_path) as f:
        card = json.load(f)

    md = render_article_card(card)
    md_path.write_text(md, encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an Article Literature Card JSON as Markdown."
    )
    parser.add_argument("--article-id", required=True, help="Article folder name")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing article_card.md")
    args = parser.parse_args()

    md_path = render_article_card_file(args.article_id, overwrite=args.overwrite)
    print(f"Written: {md_path}")


if __name__ == "__main__":
    main()
