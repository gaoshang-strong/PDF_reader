"""Tests for pipeline/step1_canonical_articles.py"""

import json
import math
import textwrap
from pathlib import Path

import pytest

from pipeline.step1_canonical_articles import (
    compute_report,
    copy_images,
    copy_raw_markdown,
    discover_article_ids,
    extract_grobid_metadata,
    process_article,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_TEI = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <teiHeader>
        <fileDesc>
          <titleStmt>
            <title level="a" type="main">A Great Paper</title>
          </titleStmt>
          <publicationStmt><publisher/></publicationStmt>
          <sourceDesc>
            <biblStruct>
              <analytic>
                <author>
                  <persName>
                    <forename type="first">Alice</forename>
                    <surname>Smith</surname>
                  </persName>
                  <affiliation key="aff0">
                    <orgName type="institution">University of Testing</orgName>
                    <settlement>Testville</settlement>
                    <country key="US">USA</country>
                  </affiliation>
                </author>
                <author>
                  <persName>
                    <forename type="first">Bob</forename>
                    <surname>Jones</surname>
                  </persName>
                  <affiliation key="aff0"/>
                </author>
              </analytic>
              <monogr>
                <title level="j">Journal of Tests</title>
                <idno type="DOI">10.9999/test.2024.001</idno>
                <imprint>
                  <date type="published" when="2024-03-15"/>
                </imprint>
              </monogr>
            </biblStruct>
          </sourceDesc>
        </fileDesc>
        <profileDesc>
          <abstract>
            <p>This is the abstract text.</p>
          </abstract>
        </profileDesc>
      </teiHeader>
      <text>
        <back>
          <div type="references">
            <listBibl>
              <biblStruct>
                <analytic>
                  <title type="main">Referenced Work</title>
                  <author>
                    <persName>
                      <forename type="first">Carol</forename>
                      <surname>White</surname>
                    </persName>
                  </author>
                </analytic>
                <monogr>
                  <title level="j">Ref Journal</title>
                  <imprint>
                    <date type="published" when="2020"/>
                  </imprint>
                </monogr>
                <idno type="DOI">10.1234/ref.2020</idno>
              </biblStruct>
            </listBibl>
          </div>
        </back>
      </text>
    </TEI>
""")


def _make_mineru(data_root: Path, article_id: str, md_content: str = "Hello world") -> None:
    base = data_root / "parser_outputs" / "mineru_work" / article_id / article_id
    base.mkdir(parents=True)
    (base / "full.md").write_text(md_content, encoding="utf-8")


def _make_images(data_root: Path, article_id: str) -> None:
    img_dir = (
        data_root / "parser_outputs" / "mineru_work" / article_id / article_id / "images"
    )
    img_dir.mkdir(parents=True)
    (img_dir / "fig1.jpg").write_bytes(b"\xff\xd8\xff")
    (img_dir / "fig2.png").write_bytes(b"\x89PNG")


def _make_tei(data_root: Path, article_id: str, content: str = _MINIMAL_TEI) -> None:
    tei_dir = data_root / "parser_outputs" / "grobid_tei"
    tei_dir.mkdir(parents=True, exist_ok=True)
    (tei_dir / f"{article_id}.tei.xml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Article discovery
# ---------------------------------------------------------------------------

def test_discover_article_ids_returns_sorted(tmp_path):
    for aid in ("pdf_ccc", "pdf_aaa", "pdf_bbb"):
        (tmp_path / "parser_outputs" / "mineru_work" / aid).mkdir(parents=True)
    ids = discover_article_ids(tmp_path)
    assert ids == ["pdf_aaa", "pdf_bbb", "pdf_ccc"]


def test_discover_article_ids_empty(tmp_path):
    (tmp_path / "parser_outputs" / "mineru_work").mkdir(parents=True)
    assert discover_article_ids(tmp_path) == []


def test_discover_article_ids_missing_dir(tmp_path):
    assert discover_article_ids(tmp_path) == []


# ---------------------------------------------------------------------------
# Raw markdown copying
# ---------------------------------------------------------------------------

def test_copy_raw_markdown_creates_file(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="# Title\n\nBody text here.")
    dst = copy_raw_markdown("pdf_abc", tmp_path)
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "# Title\n\nBody text here."


def test_copy_raw_markdown_preserves_content_exactly(tmp_path):
    original = "line1\nline2\n\n## Section\n\n- item\n"
    _make_mineru(tmp_path, "pdf_abc", md_content=original)
    dst = copy_raw_markdown("pdf_abc", tmp_path)
    assert dst.read_text(encoding="utf-8") == original


def test_copy_raw_markdown_no_overwrite_by_default(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="original")
    copy_raw_markdown("pdf_abc", tmp_path)
    # overwrite source
    (tmp_path / "parser_outputs" / "mineru_work" / "pdf_abc" / "pdf_abc" / "full.md").write_text(
        "changed", encoding="utf-8"
    )
    copy_raw_markdown("pdf_abc", tmp_path, overwrite=False)
    dst = tmp_path / "articles" / "pdf_abc" / "article_text.raw.md"
    assert dst.read_text(encoding="utf-8") == "original"


def test_copy_raw_markdown_overwrite(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="original")
    copy_raw_markdown("pdf_abc", tmp_path)
    (tmp_path / "parser_outputs" / "mineru_work" / "pdf_abc" / "pdf_abc" / "full.md").write_text(
        "changed", encoding="utf-8"
    )
    copy_raw_markdown("pdf_abc", tmp_path, overwrite=True)
    dst = tmp_path / "articles" / "pdf_abc" / "article_text.raw.md"
    assert dst.read_text(encoding="utf-8") == "changed"


def test_copy_raw_markdown_output_path(tmp_path):
    _make_mineru(tmp_path, "pdf_xyz")
    dst = copy_raw_markdown("pdf_xyz", tmp_path)
    assert dst == tmp_path / "articles" / "pdf_xyz" / "article_text.raw.md"


# ---------------------------------------------------------------------------
# Image copying
# ---------------------------------------------------------------------------

def test_copy_images_copies_directory(tmp_path):
    _make_mineru(tmp_path, "pdf_abc")
    _make_images(tmp_path, "pdf_abc")
    dst = copy_images("pdf_abc", tmp_path)
    assert dst is not None
    assert (dst / "fig1.jpg").exists()
    assert (dst / "fig2.png").exists()


def test_copy_images_returns_none_when_missing(tmp_path):
    _make_mineru(tmp_path, "pdf_abc")
    result = copy_images("pdf_abc", tmp_path)
    assert result is None


def test_copy_images_no_overwrite_by_default(tmp_path):
    _make_mineru(tmp_path, "pdf_abc")
    _make_images(tmp_path, "pdf_abc")
    copy_images("pdf_abc", tmp_path)
    # add a sentinel in dst to verify it's not replaced
    sentinel = tmp_path / "articles" / "pdf_abc" / "images" / "sentinel.txt"
    sentinel.write_text("keep me")
    copy_images("pdf_abc", tmp_path, overwrite=False)
    assert sentinel.exists()


def test_copy_images_overwrite_replaces_directory(tmp_path):
    _make_mineru(tmp_path, "pdf_abc")
    _make_images(tmp_path, "pdf_abc")
    copy_images("pdf_abc", tmp_path)
    sentinel = tmp_path / "articles" / "pdf_abc" / "images" / "sentinel.txt"
    sentinel.write_text("old file")
    copy_images("pdf_abc", tmp_path, overwrite=True)
    assert not sentinel.exists()
    assert (tmp_path / "articles" / "pdf_abc" / "images" / "fig1.jpg").exists()


# ---------------------------------------------------------------------------
# GROBID metadata extraction
# ---------------------------------------------------------------------------

def test_extract_metadata_title(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["title"] == "A Great Paper"


def test_extract_metadata_authors(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["authors"] == ["Alice Smith", "Bob Jones"]


def test_extract_metadata_authors_deduplicated(tmp_path):
    # Bob Jones appears twice via two author elements with same name
    tei = _MINIMAL_TEI.replace(
        "</analytic>",
        """  <author>
                  <persName><forename type="first">Bob</forename><surname>Jones</surname></persName>
                </author>
              </analytic>""",
        1,
    )
    _make_tei(tmp_path, "pdf_abc", tei)
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["authors"].count("Bob Jones") == 1


def test_extract_metadata_affiliations(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert len(meta["affiliations"]) == 1
    assert "University of Testing" in meta["affiliations"][0]


def test_extract_metadata_doi(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["DOI"] == "10.9999/test.2024.001"


def test_extract_metadata_publication_year(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["publication_year"] == 2024


def test_extract_metadata_abstract(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["abstract"] is not None
    assert "abstract text" in meta["abstract"]


def test_extract_metadata_references(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert len(meta["references"]) == 1
    ref = meta["references"][0]
    assert ref["title"] == "Referenced Work"
    assert ref["journal"] == "Ref Journal"
    assert ref["doi"] == "10.1234/ref.2020"
    assert ref["year"] == 2020
    assert ref["authors"] == ["Carol White"]


def test_extract_metadata_missing_tei(tmp_path):
    meta = extract_grobid_metadata("pdf_missing", tmp_path)
    assert meta["article_id"] == "pdf_missing"
    assert meta["title"] is None
    assert meta["authors"] == []
    assert meta["affiliations"] == []
    assert meta["journal"] is None
    assert meta["DOI"] is None
    assert meta["publication_year"] is None
    assert meta["abstract"] is None
    assert meta["keywords"] == []
    assert meta["references"] == []


def test_extract_metadata_empty_fields_are_null_or_list(tmp_path):
    tei_no_keywords = _MINIMAL_TEI  # our minimal TEI has no <keywords>
    _make_tei(tmp_path, "pdf_abc", tei_no_keywords)
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["keywords"] == []
    assert isinstance(meta["references"], list)


def test_extract_metadata_article_id_always_present(tmp_path):
    _make_tei(tmp_path, "pdf_abc")
    meta = extract_grobid_metadata("pdf_abc", tmp_path)
    assert meta["article_id"] == "pdf_abc"


# ---------------------------------------------------------------------------
# Raw article report
# ---------------------------------------------------------------------------

def _make_article_md(data_root: Path, article_id: str, text: str) -> None:
    dst = data_root / "articles" / article_id
    dst.mkdir(parents=True)
    (dst / "article_text.raw.md").write_text(text, encoding="utf-8")


def test_compute_report_fields(tmp_path):
    text = "hello world"
    _make_article_md(tmp_path, "pdf_abc", text)
    report = compute_report("pdf_abc", tmp_path)
    assert report["article_id"] == "pdf_abc"
    assert report["raw_char_count"] == len(text)
    assert report["raw_word_count"] == 2
    assert report["estimated_token_count"] == math.ceil(len(text) / 4)


def test_compute_report_pass(tmp_path):
    # 90000 * 4 = 360000 chars → exactly PASS boundary
    text = "a" * (90_000 * 4)
    _make_article_md(tmp_path, "pdf_abc", text)
    assert compute_report("pdf_abc", tmp_path)["length_status"] == "PASS"


def test_compute_report_warn_lower(tmp_path):
    # token count = 90001
    text = "a" * (90_001 * 4)
    _make_article_md(tmp_path, "pdf_abc", text)
    assert compute_report("pdf_abc", tmp_path)["length_status"] == "WARN"


def test_compute_report_warn_upper(tmp_path):
    # token count = 115000 → still WARN
    text = "a" * (115_000 * 4)
    _make_article_md(tmp_path, "pdf_abc", text)
    assert compute_report("pdf_abc", tmp_path)["length_status"] == "WARN"


def test_compute_report_fail(tmp_path):
    # token count = 115001 → FAIL
    text = "a" * (115_001 * 4)
    _make_article_md(tmp_path, "pdf_abc", text)
    assert compute_report("pdf_abc", tmp_path)["length_status"] == "FAIL"


def test_compute_report_token_count_ceiling(tmp_path):
    # 5 chars → ceil(5/4) = 2
    _make_article_md(tmp_path, "pdf_abc", "abcde")
    assert compute_report("pdf_abc", tmp_path)["estimated_token_count"] == 2


# ---------------------------------------------------------------------------
# process_article integration
# ---------------------------------------------------------------------------

def test_process_article_creates_all_outputs(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="short article text")
    _make_images(tmp_path, "pdf_abc")
    _make_tei(tmp_path, "pdf_abc")
    process_article("pdf_abc", tmp_path)
    base = tmp_path / "articles" / "pdf_abc"
    assert (base / "article_text.raw.md").exists()
    assert (base / "images").is_dir()
    assert (base / "metadata.json").exists()
    assert (base / "raw_article_report.json").exists()


def test_process_article_metadata_json_valid(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="text")
    _make_tei(tmp_path, "pdf_abc")
    process_article("pdf_abc", tmp_path)
    meta = json.loads((tmp_path / "articles" / "pdf_abc" / "metadata.json").read_text())
    assert meta["article_id"] == "pdf_abc"
    assert meta["title"] == "A Great Paper"


def test_process_article_report_json_valid(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="short text")
    _make_tei(tmp_path, "pdf_abc")
    process_article("pdf_abc", tmp_path)
    report = json.loads((tmp_path / "articles" / "pdf_abc" / "raw_article_report.json").read_text())
    assert report["article_id"] == "pdf_abc"
    assert "length_status" in report
    assert report["length_status"] == "PASS"


def test_process_article_no_images_no_error(tmp_path):
    _make_mineru(tmp_path, "pdf_abc")
    _make_tei(tmp_path, "pdf_abc")
    process_article("pdf_abc", tmp_path)
    assert not (tmp_path / "articles" / "pdf_abc" / "images").exists()


def test_process_article_skips_existing_without_overwrite(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="original")
    _make_tei(tmp_path, "pdf_abc")
    process_article("pdf_abc", tmp_path)
    # Write a sentinel into the output files
    meta_path = tmp_path / "articles" / "pdf_abc" / "metadata.json"
    meta_path.write_text('{"article_id": "sentinel"}', encoding="utf-8")
    process_article("pdf_abc", tmp_path, overwrite=False)
    assert json.loads(meta_path.read_text())["article_id"] == "sentinel"


def test_process_article_overwrites_when_flag_set(tmp_path):
    _make_mineru(tmp_path, "pdf_abc", md_content="original")
    _make_tei(tmp_path, "pdf_abc")
    process_article("pdf_abc", tmp_path)
    meta_path = tmp_path / "articles" / "pdf_abc" / "metadata.json"
    meta_path.write_text('{"article_id": "sentinel"}', encoding="utf-8")
    process_article("pdf_abc", tmp_path, overwrite=True)
    assert json.loads(meta_path.read_text())["article_id"] == "pdf_abc"


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------

def _setup_two_articles(tmp_path: Path) -> list[str]:
    for aid in ("pdf_aaa", "pdf_bbb"):
        _make_mineru(tmp_path, aid, md_content="sample text for " + aid)
        _make_tei(tmp_path, aid)
    return ["pdf_aaa", "pdf_bbb"]


def test_cli_all_articles(tmp_path):
    from pipeline.step1_canonical_articles import main as step1_main

    aids = _setup_two_articles(tmp_path)
    import sys

    sys.argv = ["step1", "--data-root", str(tmp_path)]
    step1_main()
    for aid in aids:
        assert (tmp_path / "articles" / aid / "article_text.raw.md").exists()
        assert (tmp_path / "articles" / aid / "metadata.json").exists()
        assert (tmp_path / "articles" / aid / "raw_article_report.json").exists()


def test_cli_single_article(tmp_path):
    from pipeline.step1_canonical_articles import main as step1_main

    _setup_two_articles(tmp_path)
    import sys

    sys.argv = ["step1", "--data-root", str(tmp_path), "--article-id", "pdf_aaa"]
    step1_main()
    assert (tmp_path / "articles" / "pdf_aaa" / "article_text.raw.md").exists()
    assert not (tmp_path / "articles" / "pdf_bbb").exists()


def test_cli_overwrite_flag(tmp_path):
    from pipeline.step1_canonical_articles import main as step1_main

    _make_mineru(tmp_path, "pdf_aaa", md_content="original")
    _make_tei(tmp_path, "pdf_aaa")
    import sys

    sys.argv = ["step1", "--data-root", str(tmp_path), "--article-id", "pdf_aaa"]
    step1_main()

    # Mutate the output
    meta_path = tmp_path / "articles" / "pdf_aaa" / "metadata.json"
    meta_path.write_text('{"article_id": "sentinel"}', encoding="utf-8")

    # Run without --overwrite — sentinel should survive
    step1_main()
    assert json.loads(meta_path.read_text())["article_id"] == "sentinel"

    # Run with --overwrite — sentinel should be gone
    sys.argv = ["step1", "--data-root", str(tmp_path), "--article-id", "pdf_aaa", "--overwrite"]
    step1_main()
    assert json.loads(meta_path.read_text())["article_id"] == "pdf_aaa"
