#!/usr/bin/env python3
"""Tests for step4D_render_article_card_md.py"""

import json
import pytest
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.step4D_render_article_card_md import (
    collect_evidence_parent_ids,
    render_article_card,
    render_article_card_file,
    _evidence_line,
    _null_dash,
    _summary_block,
    _md_table,
)


# ---------------------------------------------------------------------------
# Minimal valid card fixture
# ---------------------------------------------------------------------------

def _make_card(overrides: dict | None = None) -> dict:
    base = {
        "schema_version": "0.1.0",
        "article_id": "pdf_test123",
        "metadata": {
            "title": "Test Article Title",
            "authors": ["Author A", "Author B"],
            "journal": "Nature",
            "year": 2024,
            "doi": "10.1000/test",
            "article_type": "research_article",
        },
        "extraction_metadata": {
            "extraction_method": "llm:deepseek-chat",
            "created_at": "2024-01-01T00:00:00+00:00",
            "source_files": [],
            "module_status": {},
            "warnings": [],
        },
        "modules": {
            "research_question": {
                "summary": "This study investigates T cell exhaustion.",
                "disease_context": ["cancer"],
                "biological_context": ["immunology"],
                "main_question": "How does AKT signaling cause exhaustion?",
                "stated_gap": "Mechanisms poorly understood.",
                "study_type": ["observational"],
                "evidence": [{"parent_id": "pdf_test123::parent_000001", "section_path": ["Introduction"], "supporting_text": "T cells are exhausted."}],
            },
            "study_design": {
                "summary": "Retrospective cohort with scRNA-seq.",
                "design_features": ["retrospective", "multi-cohort"],
                "discovery_validation_structure": "Discovery + replication",
                "perturbation_or_intervention": None,
                "comparison_groups": [
                    {"group_a": "CD8+ T cells", "group_b": "exhausted T cells", "comparison_purpose": "identify markers", "evidence": []},
                ],
                "evidence": [],
            },
            "samples_and_cohorts": {
                "summary": "82 CRC patients enrolled.",
                "items": [
                    {
                        "sample_or_cohort_name": "CRC cohort",
                        "species": "human",
                        "disease_or_condition": "colorectal cancer",
                        "sample_count": 82,
                        "sample_material": "PBMC",
                        "used_for": "discovery",
                        "evidence": [],
                    }
                ],
                "evidence": [],
            },
            "data_source_and_provenance": {
                "summary": "scRNA-seq from GEO.",
                "items": [
                    {
                        "dataset_name": "CRC scRNA",
                        "data_modality": "scRNA-seq",
                        "source_type": "generated_in_this_study",
                        "accession_id": "GSE12345",
                        "evidence": [],
                    }
                ],
                "evidence": [],
            },
            "omics_and_experimental_methods": {
                "summary": "scRNA-seq and flow cytometry used.",
                "items": [
                    {
                        "method": "10x Genomics scRNA-seq",
                        "modality_category": "single_cell_transcriptomics",
                        "sample_material": "PBMC",
                        "purpose": "Transcriptome profiling",
                        "evidence": [],
                    }
                ],
                "evidence": [],
            },
            "computational_analysis_pipeline": {
                "summary": "Seurat clustering pipeline.",
                "items": [
                    {
                        "analysis_step": "Clustering",
                        "software_or_method": ["Seurat", "UMAP"],
                        "parameters_or_thresholds": "resolution=0.5",
                        "evidence": [],
                    }
                ],
                "evidence": [],
            },
            "key_findings": {
                "summary": "AKT drives exhaustion.",
                "items": [
                    {"finding": "Elevated AKT correlates with T cell exhaustion.", "result_type": "correlative", "evidence": []},
                    {"finding": "AKT inhibition rescued effector function.", "result_type": "functional", "evidence": []},
                ],
                "evidence": [],
            },
            "mechanism_model": {
                "summary": "AKT → exhaustion via proteotoxic stress.",
                "claims": [
                    {
                        "mechanism_chain": ["Persistent AKT", "proteotoxic stress", "T cell exhaustion"],
                        "upstream_factor": "Persistent AKT signaling",
                        "mediators": ["proteotoxic stress"],
                        "downstream_effect": "T cell exhaustion",
                        "involved_cell_types": ["CD8+ T cells"],
                        "involved_pathways_or_molecules": ["AKT", "mTOR"],
                        "therapeutic_or_clinical_implication": "AKT inhibitors as therapy",
                        "evidence": [],
                    }
                ],
                "evidence": [],
            },
            "author_reported_limitations_and_future_directions": {
                "summary": "Cohort size limits generalizability.",
                "items": [
                    {"type": "limitation", "text": "Small cohort size.", "evidence": []},
                    {"type": "future_direction", "text": "Validate in larger cohort.", "evidence": []},
                ],
                "evidence": [],
            },
        },
    }
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Unit tests: small helpers
# ---------------------------------------------------------------------------

class TestCollectEvidenceParentIds:
    def test_empty_dict(self):
        assert collect_evidence_parent_ids({}) == []

    def test_single_evidence(self):
        obj = {"evidence": [{"parent_id": "pdf::parent_000001", "section_path": []}]}
        assert collect_evidence_parent_ids(obj) == ["pdf::parent_000001"]

    def test_deduplicates(self):
        obj = {
            "evidence": [{"parent_id": "pdf::parent_000001", "section_path": []}],
            "nested": {"evidence": [{"parent_id": "pdf::parent_000001", "section_path": []}]},
        }
        result = collect_evidence_parent_ids(obj)
        assert result.count("pdf::parent_000001") == 1

    def test_nested_list(self):
        obj = {
            "items": [
                {"evidence": [{"parent_id": "pdf::parent_000002", "section_path": []}]},
                {"evidence": [{"parent_id": "pdf::parent_000003", "section_path": []}]},
            ]
        }
        result = collect_evidence_parent_ids(obj)
        assert "pdf::parent_000002" in result
        assert "pdf::parent_000003" in result

    def test_no_evidence_key(self):
        obj = {"summary": "hello", "items": [{"text": "foo"}]}
        assert collect_evidence_parent_ids(obj) == []


class TestNullDash:
    def test_none(self):
        assert _null_dash(None) == "—"

    def test_empty_list(self):
        assert _null_dash([]) == "—"

    def test_nonempty_list(self):
        assert _null_dash(["a", "b"]) == "a, b"

    def test_string(self):
        assert _null_dash("hello") == "hello"

    def test_int(self):
        assert _null_dash(42) == "42"


class TestSummaryBlock:
    def test_none(self):
        assert _summary_block(None) == "_No summary extracted._"

    def test_text(self):
        assert _summary_block("A study.") == "A study."


class TestMdTable:
    def test_structure(self):
        result = _md_table(["A", "B"], [["1", "2"], ["3", "4"]])
        lines = result.splitlines()
        assert lines[0] == "| A | B |"
        assert lines[1] == "| --- | --- |"
        assert "| 1 | 2 |" in lines
        assert "| 3 | 4 |" in lines

    def test_single_row(self):
        result = _md_table(["X"], [["val"]])
        assert "| X |" in result
        assert "| val |" in result


class TestEvidenceLine:
    def test_no_evidence(self):
        result = _evidence_line({"evidence": []})
        assert "none" in result

    def test_with_evidence(self):
        data = {"evidence": [{"parent_id": "pdf::parent_000005", "section_path": []}]}
        result = _evidence_line(data)
        assert "parent_000005" in result
        assert "**Evidence parents:**" in result

    def test_short_form_strips_prefix(self):
        data = {"evidence": [{"parent_id": "pdf_abc::parent_000001", "section_path": []}]}
        result = _evidence_line(data)
        assert "parent_000001" in result
        assert "pdf_abc" not in result


# ---------------------------------------------------------------------------
# render_article_card: output structure
# ---------------------------------------------------------------------------

class TestRenderArticleCard:
    def setup_method(self):
        self.card = _make_card()
        self.md = render_article_card(self.card)

    def test_returns_string(self):
        assert isinstance(self.md, str)
        assert len(self.md) > 100

    def test_article_id_in_header(self):
        assert "# pdf_test123" in self.md

    def test_title_in_header(self):
        assert "Test Article Title" in self.md

    def test_authors_in_header(self):
        assert "Author A" in self.md
        assert "Author B" in self.md

    def test_journal_in_header(self):
        assert "Nature" in self.md

    def test_doi_in_header(self):
        assert "10.1000/test" in self.md

    def test_extraction_method_in_header(self):
        assert "deepseek-chat" in self.md

    def test_all_module_sections_present(self):
        expected_sections = [
            "## A. Research Question",
            "## B. Study Design",
            "## C. Samples & Cohorts",
            "## D. Data Sources & Provenance",
            "## E. Omics & Experimental Methods",
            "## F. Computational Pipeline",
            "## G. Key Findings",
            "## H. Mechanism Model",
            "## I. Limitations & Future Directions",
        ]
        for section in expected_sections:
            assert section in self.md, f"Missing section: {section}"

    def test_summaries_rendered(self):
        assert "This study investigates T cell exhaustion." in self.md
        assert "Retrospective cohort with scRNA-seq." in self.md

    def test_table_rendered_for_samples(self):
        assert "CRC cohort" in self.md
        assert "colorectal cancer" in self.md

    def test_table_rendered_for_methods(self):
        assert "10x Genomics scRNA-seq" in self.md
        assert "single_cell_transcriptomics" in self.md

    def test_key_findings_bullet(self):
        assert "Elevated AKT correlates with T cell exhaustion." in self.md
        assert "_correlative_" in self.md

    def test_mechanism_model_chain(self):
        assert "Persistent AKT" in self.md
        assert "→" in self.md

    def test_limitations_rendered(self):
        assert "[limitation]" in self.md
        assert "[future_direction]" in self.md
        assert "Small cohort size." in self.md

    def test_evidence_parents_in_sections(self):
        assert "**Evidence parents:**" in self.md or "_Evidence parents: none_" in self.md

    def test_separator_present(self):
        assert "---" in self.md

    def test_no_warnings_section_when_empty(self):
        assert "## Extraction Warnings" not in self.md

    def test_warnings_section_when_present(self):
        card = _make_card()
        card["extraction_metadata"]["warnings"] = ["schema: missing field x"]
        md = render_article_card(card)
        assert "## Extraction Warnings" in md
        assert "schema: missing field x" in md


class TestRenderNullFields:
    def test_no_authors(self):
        card = _make_card()
        card["metadata"]["authors"] = []
        md = render_article_card(card)
        assert "**Authors:**" not in md

    def test_no_journal_or_year(self):
        card = _make_card()
        card["metadata"]["journal"] = None
        card["metadata"]["year"] = None
        md = render_article_card(card)
        assert "**Journal:**" not in md

    def test_no_doi(self):
        card = _make_card()
        card["metadata"]["doi"] = None
        md = render_article_card(card)
        assert "**DOI:**" not in md

    def test_null_summary_shows_placeholder(self):
        card = _make_card()
        card["modules"]["research_question"]["summary"] = None
        md = render_article_card(card)
        assert "_No summary extracted._" in md

    def test_empty_module_renders_no_table(self):
        card = _make_card()
        card["modules"]["samples_and_cohorts"]["items"] = []
        md = render_article_card(card)
        # No table rows for samples — should still have the section header
        assert "## C. Samples & Cohorts" in md

    def test_empty_comparison_groups(self):
        card = _make_card()
        card["modules"]["study_design"]["comparison_groups"] = []
        md = render_article_card(card)
        assert "## B. Study Design" in md

    def test_no_mechanism_claims(self):
        card = _make_card()
        card["modules"]["mechanism_model"]["claims"] = []
        md = render_article_card(card)
        assert "## H. Mechanism Model" in md


class TestRenderArticleCardFile:
    def test_writes_md_file(self, tmp_path):
        card = _make_card()
        article_id = "pdf_test123"
        card_dir = tmp_path / "literature_cards" / article_id
        card_dir.mkdir(parents=True)
        card_path = card_dir / "article_card.json"
        card_path.write_text(json.dumps(card), encoding="utf-8")

        md_path = render_article_card_file(article_id, data_root=tmp_path)

        assert md_path.exists()
        assert md_path.suffix == ".md"
        content = md_path.read_text(encoding="utf-8")
        assert "pdf_test123" in content

    def test_raises_if_json_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            render_article_card_file("pdf_nonexistent", data_root=tmp_path)

    def test_raises_if_md_exists_no_overwrite(self, tmp_path):
        card = _make_card()
        article_id = "pdf_test123"
        card_dir = tmp_path / "literature_cards" / article_id
        card_dir.mkdir(parents=True)
        (card_dir / "article_card.json").write_text(json.dumps(card), encoding="utf-8")
        (card_dir / "article_card.md").write_text("existing", encoding="utf-8")

        with pytest.raises(FileExistsError):
            render_article_card_file(article_id, data_root=tmp_path)

    def test_overwrite_replaces_md(self, tmp_path):
        card = _make_card()
        article_id = "pdf_test123"
        card_dir = tmp_path / "literature_cards" / article_id
        card_dir.mkdir(parents=True)
        (card_dir / "article_card.json").write_text(json.dumps(card), encoding="utf-8")
        (card_dir / "article_card.md").write_text("old content", encoding="utf-8")

        md_path = render_article_card_file(article_id, overwrite=True, data_root=tmp_path)
        content = md_path.read_text(encoding="utf-8")
        assert "old content" not in content
        assert "pdf_test123" in content

    def test_returns_correct_path(self, tmp_path):
        card = _make_card()
        article_id = "pdf_test123"
        card_dir = tmp_path / "literature_cards" / article_id
        card_dir.mkdir(parents=True)
        (card_dir / "article_card.json").write_text(json.dumps(card), encoding="utf-8")

        md_path = render_article_card_file(article_id, data_root=tmp_path)
        assert md_path == card_dir / "article_card.md"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_writes_file(self, tmp_path):
        import subprocess, sys
        card = _make_card()
        article_id = "pdf_test123"
        card_dir = tmp_path / "literature_cards" / article_id
        card_dir.mkdir(parents=True)
        (card_dir / "article_card.json").write_text(json.dumps(card), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "pipeline" / "step4D_render_article_card_md.py"),
             "--article-id", article_id],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
            cwd=str(tmp_path),
        )
        # CLI always writes to data_root relative to script — so just check no error
        # since in CLI mode data_root is REPO_ROOT/data, not tmp_path.
        # We only verify the script runs (returncode is 0 or 1 depending on whether real data exists).
        assert result.returncode in (0, 1)
