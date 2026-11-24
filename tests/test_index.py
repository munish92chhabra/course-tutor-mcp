"""Known-answer tests for the course index.

These are deliberately written the way a bench scientist writes controls:
each one asserts something whose answer is known independently of the code,
so a passing test means the retrieval is actually working rather than
merely running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notes_index import (  # noqa: E402
    CourseLibrary,
    split_markdown,
    split_questions,
    tokenize,
    topic_from_filename,
)

COURSE_DIR = Path(__file__).resolve().parents[1] / "course"


@pytest.fixture(scope="module")
def library() -> CourseLibrary:
    return CourseLibrary(COURSE_DIR)


# --- loading ---------------------------------------------------------------


def test_loads_both_kinds(library):
    assert library.of_kind("notes"), "no lecture notes were indexed"
    assert library.of_kind("assessment"), "no past questions were indexed"


def test_reads_markdown_and_pdf(library):
    suffixes = {Path(s.source).suffix.lower() for s in library.of_kind("notes")}
    assert ".md" in suffixes, "markdown notes were not indexed"
    assert ".pdf" in suffixes, "PDF notes were not indexed"


def test_pdf_sections_cite_a_page(library):
    pdf_sections = [
        s for s in library.of_kind("notes") if s.source.lower().endswith(".pdf")
    ]
    assert pdf_sections
    assert all(s.locator.startswith("p. ") for s in pdf_sections)


def test_long_pdf_page_is_split_into_parts(library):
    locators = [s.locator for s in library.of_kind("notes")]
    # The sample lecture PDF has dense prose pages, so at least one page
    # must have been broken into parts rather than indexed whole.
    assert any("(part " in l for l in locators), locators


def test_no_empty_sections(library):
    assert all(s.text.strip() for s in library.sections)


# --- question splitting ----------------------------------------------------


def test_exam_split_into_questions(library):
    questions = library.of_kind("assessment")
    locators = {q.locator for q in questions}
    # The sample paper has seven numbered questions.
    assert {"Q1", "Q4", "Q7"} <= locators, locators


def test_marks_are_extracted(library):
    q3 = next(q for q in library.of_kind("assessment") if q.locator == "Q3")
    assert q3.marks == 5


def test_prose_is_not_mistaken_for_a_question_paper():
    prose = "Intro line.\n\n1. first item\n\nSome following prose.\n"
    assert split_questions(prose) == []


# --- search ----------------------------------------------------------------


def test_search_finds_the_right_lecture_section(library):
    hits = library.search("why is the lag phase not dormant", top_k=3, kind="notes")
    assert hits, "no hit for a question the notes clearly answer"
    text = hits[0][0].text.lower()
    assert "lag" in text


def test_search_separates_notes_from_assessments(library):
    note_hits = library.search("growth curve", top_k=5, kind="notes")
    exam_hits = library.search("growth curve", top_k=5, kind="assessment")
    assert all(s.kind == "notes" for s, _ in note_hits)
    assert all(s.kind == "assessment" for s, _ in exam_hits)


def test_past_question_lookup_finds_the_calculation(library):
    hits = library.search("generation time calculation", top_k=3, kind="assessment")
    assert hits
    assert "generation" in hits[0][0].text.lower()


def test_absent_topic_returns_nothing(library):
    # Not covered anywhere in the sample course.
    assert library.search("quantum chromodynamics", top_k=5) == []


def test_ranking_prefers_the_more_relevant_section(library):
    hits = library.search("crystal violet decolourising", top_k=5, kind="notes")
    assert hits
    assert "crystal violet" in hits[0][0].text.lower()


# --- retrieval by locator --------------------------------------------------


def test_get_section_roundtrip(library):
    first = library.of_kind("assessment")[0]
    got = library.get(first.source, first.locator)
    assert got is not None and got.text == first.text


def test_get_section_is_case_insensitive(library):
    first = library.of_kind("assessment")[0]
    assert library.get(first.source.upper(), first.locator.lower()) is not None


def test_get_section_missing_returns_none(library):
    assert library.get("nonexistent.pdf", "Q99") is None


# --- helpers ---------------------------------------------------------------


def test_topic_from_filename_strips_numbering_and_prefix():
    assert topic_from_filename(Path("02_growth_and_culture.md")) == "Growth And Culture"
    assert topic_from_filename(Path("lecture_03_staining.pdf")) == "Staining"
    assert topic_from_filename(Path("sample_midterm.pdf")) == "Sample Midterm"


def test_tokenize_drops_stopwords():
    assert "the" not in tokenize("the bacterial cell wall")
    assert "bacterial" in tokenize("the bacterial cell wall")


def test_split_markdown_uses_headings():
    pairs = split_markdown("# Title\n\nbody one\n\n## Second\n\nbody two\n")
    headings = [h for h, _ in pairs]
    assert "Title" in headings and "Second" in headings
