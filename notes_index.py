"""Load, chunk and search a course's own materials.

Three kinds of material are handled, and the differences matter:

  notes/               lecture notes - the material to explain *from*
  assessments/         past papers - the questions to model new ones *on*
  assessments_private/ marking schemes and model answers - TEACHER ONLY

Keeping notes and past papers apart is what lets the model say "this was
asked in the 2026 midterm" and write new questions in the style students
will actually face, instead of blending everything into one pile of text.

The third folder is separated for a different reason: safety. Marking
schemes must never reach a student, so they are loaded as their own kind
and the web app scopes retrieval by role. A student session cannot retrieve
them even if a student asks for them directly, because the search is never
run over that kind - it is not a matter of the model declining.

The search is a compact BM25 implementation rather than an embedding model,
because it runs offline with no API key, it is fast enough for a course-sized
corpus, and the scores are explainable. For a much larger corpus, swap
`BM25Index` for an embedding store; the `search()` interface stays the same.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Kind = Literal["notes", "assessment", "marking_scheme"]

#: Kinds a student may ever retrieve. Marking schemes are deliberately absent.
STUDENT_KINDS: tuple[Kind, ...] = ("notes", "assessment")
TEACHER_KINDS: tuple[Kind, ...] = ("notes", "assessment", "marking_scheme")

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
READABLE_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}

#: Target size for a chunk from an unstructured text file, in words.
TARGET_CHUNK_WORDS = 220

#: A PDF page shorter than this is merged into the next one, so that a
#: title slide or a near-empty page is not indexed as its own section.
MIN_PAGE_WORDS = 25

#: A PDF page longer than this is split into parts. A slide makes a good
#: chunk on its own; a dense page of prose does not, because retrieving it
#: whole buries the relevant paragraph in several hundred words of context.
MAX_PAGE_WORDS = 400


# --------------------------------------------------------------------------
# What a retrievable piece of material looks like
# --------------------------------------------------------------------------


@dataclass
class Section:
    """One retrievable piece of course material."""

    source: str  # file name, e.g. "lecture_02_growth.pdf"
    locator: str  # heading, or "p. 4", or "Q3"
    text: str
    topic: str  # derived from the file name
    kind: Kind  # "notes" or "assessment"
    marks: int | None = None  # for exam questions, if the paper states them
    tokens: list[str] = field(default_factory=list, repr=False)

    @property
    def citation(self) -> str:
        return f"{self.source}, {self.locator}"

    def preview(self, words: int = 60) -> str:
        parts = self.text.split()
        if len(parts) <= words:
            return self.text
        return " ".join(parts[:words]) + " ..."


def topic_from_filename(path: Path) -> str:
    """Turn a file name into a readable topic.

    'lecture_02_growth_and_culture.pdf' -> 'Growth And Culture'
    'sample_midterm.pdf'                -> 'Sample Midterm'

    Order matters: strip the prefix first, then the lecture number, or
    'lecture_02_x' keeps its 02. Exam and quiz words are deliberately kept,
    because for a past paper the whole file name is the useful label.
    """
    stem = re.sub(r"(?i)^(lecture|lec|notes?|handout)[-_\s]*", "", path.stem)
    stem = re.sub(r"^\d+[-_\s]*", "", stem)
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else path.stem.title()


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------


def read_pages(path: Path) -> list[tuple[str, str]]:
    """Read a file as (locator, text) pairs.

    PDFs come back one entry per page, labelled "p. N", so that citations
    point a student at a page they can actually turn to. Everything else
    comes back as a single entry to be chunked later.
    """
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return [("", path.read_text(encoding="utf-8", errors="replace"))]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                f"{path.name} is a PDF. Install the reader with `pip install pypdf`."
            ) from exc

        pages: list[tuple[str, str]] = []
        carry = ""
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            text = carry + ("\n\n" if carry else "") + text
            # Merge a very short page into the next one.
            if len(text.split()) < MIN_PAGE_WORDS:
                carry = text
                continue
            carry = ""
            pages.append((f"p. {i}", text))
        if carry:
            pages.append((f"p. {len(pages) + 1}", carry))
        return pages

    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                f"{path.name} is a Word file. Install the reader with "
                "`pip install python-docx`."
            ) from exc
        return [("", "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs))]

    raise RuntimeError(f"Unsupported file type: {path.name}")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def split_markdown(text: str) -> list[tuple[str, str]]:
    """Split on ATX headings. Returns (heading, body) pairs."""
    sections: list[tuple[str, list[str]]] = []
    heading = "Introduction"
    buf: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if any(l.strip() for l in buf):
                sections.append((heading, buf))
            heading = m.group(2).strip()
            buf = []
        else:
            buf.append(line)

    if any(l.strip() for l in buf):
        sections.append((heading, buf))

    return [(h, "\n".join(b).strip()) for h, b in sections]


def split_plain(text: str, label_prefix: str = "Part") -> list[tuple[str, str]]:
    """Group paragraphs into chunks of roughly TARGET_CHUNK_WORDS words."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    count = 0

    for para in paragraphs:
        buf.append(para)
        count += len(para.split())
        if count >= TARGET_CHUNK_WORDS:
            chunks.append((f"{label_prefix} {len(chunks) + 1}", "\n\n".join(buf)))
            buf, count = [], 0

    if buf:
        chunks.append((f"{label_prefix} {len(chunks) + 1}", "\n\n".join(buf)))

    return chunks


#: Line starts that look like the beginning of an exam or quiz question.
QUESTION_RE = re.compile(
    r"^\s*(?:"
    r"Q\s*\.?\s*(\d{1,2})"  # Q1  Q.1  Q 1
    r"|Question\s+(\d{1,2})"  # Question 1
    r"|(\d{1,2})\s*[.)]"  # 1.  1)
    r"|\(\s*(\d{1,2})\s*\)"  # (1)
    r")(?=[\s.):\]]|$)",  # allow the trailing punctuation of "Q1." or "Q1)"
    re.IGNORECASE,
)

MARKS_RE = re.compile(r"[\[(]\s*(\d{1,3})\s*marks?\s*[\])]", re.IGNORECASE)


def split_questions(text: str) -> list[tuple[str, str, int | None]]:
    """Split an exam or quiz into individual questions.

    Returns (locator, text, marks) triples. Falls back to an empty list if
    the document does not look like a numbered question paper, so the
    caller can chunk it as ordinary prose instead.
    """
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        m = QUESTION_RE.match(line)
        if m:
            number = next(g for g in m.groups() if g)
            starts.append((i, f"Q{number}"))

    # Require a few matches before trusting the pattern - otherwise a
    # numbered list inside ordinary prose would be mistaken for a paper.
    if len(starts) < 3:
        return []

    questions: list[tuple[str, str, int | None]] = []
    for idx, (line_no, locator) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[line_no:end]).strip()
        if not body:
            continue
        marks_match = MARKS_RE.search(body)
        marks = int(marks_match.group(1)) if marks_match else None
        questions.append((locator, body, marks))

    return questions


def sections_from_file(path: Path, kind: Kind) -> list[Section]:
    """Read one file and split it into Sections."""
    topic = topic_from_filename(path)
    pages = read_pages(path)
    out: list[Section] = []

    # Papers and marking schemes: split into individual questions first,
    # so a scheme's Q3 can be retrieved alongside the paper's Q3.
    if kind in ("assessment", "marking_scheme"):
        whole = "\n\n".join(text for _, text in pages)
        questions = split_questions(whole)
        if questions:
            return [
                Section(
                    source=path.name,
                    locator=locator,
                    text=body,
                    topic=topic,
                    kind=kind,
                    marks=marks,
                )
                for locator, body, marks in questions
            ]

    for page_locator, text in pages:
        if page_locator:  # a PDF page - cite the page it came from
            if len(text.split()) <= MAX_PAGE_WORDS:
                parts = [(page_locator, text.strip())]
            else:
                parts = [
                    (f"{page_locator} (part {i})", body)
                    for i, (_, body) in enumerate(split_plain(text), start=1)
                ]
            for locator, body in parts:
                if body.strip():
                    out.append(
                        Section(
                            source=path.name,
                            locator=locator,
                            text=body.strip(),
                            topic=topic,
                            kind=kind,
                        )
                    )
            continue

        pairs = (
            split_markdown(text)
            if path.suffix.lower() in {".md", ".markdown"}
            else split_plain(text)
        )
        for locator, body in pairs:
            if body.strip():
                out.append(
                    Section(
                        source=path.name,
                        locator=locator,
                        text=body.strip(),
                        topic=topic,
                        kind=kind,
                    )
                )

    return out


def load_folder(folder: Path, kind: Kind) -> tuple[list[Section], list[str]]:
    """Load every readable file in a folder. Returns (sections, warnings)."""
    sections: list[Section] = []
    warnings: list[str] = []

    if not folder.is_dir():
        return sections, [f"No {folder.name}/ folder at {folder.parent}"]

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in READABLE_SUFFIXES:
            continue
        try:
            found = sections_from_file(path, kind)
        except Exception as exc:  # a damaged file should not kill the server
            warnings.append(f"Could not read {path.name}: {exc}")
            continue
        if not found:
            warnings.append(
                f"{path.name} produced no text - if it is a scanned PDF it "
                "needs OCR before it can be searched."
            )
        sections.extend(found)

    return sections, warnings


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "which", "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over a list of Sections.

    BM25 scores a section by how often the query terms appear in it, damped
    so a term appearing ten times is not ten times better than once, and
    weighted so rare terms count for more than common ones. It normalises
    for length, so a long section does not win just by being long.
    """

    def __init__(self, sections: list[Section], k1: float = 1.5, b: float = 0.75):
        self.sections = sections
        self.k1 = k1
        self.b = b

        for section in self.sections:
            section.tokens = tokenize(f"{section.topic} {section.locator} {section.text}")

        self.doc_len = [len(s.tokens) for s in self.sections]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.freqs = [Counter(s.tokens) for s in self.sections]

        df: Counter[str] = Counter()
        for freq in self.freqs:
            df.update(freq.keys())

        n = len(self.sections)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        kind: Kind | tuple[Kind, ...] | None = None,
    ) -> list[tuple[Section, float]]:
        terms = tokenize(query)
        if not terms or not self.sections:
            return []

        allowed = (kind,) if isinstance(kind, str) else kind

        scored: list[tuple[Section, float]] = []
        for i, section in enumerate(self.sections):
            if allowed is not None and section.kind not in allowed:
                continue
            freq = self.freqs[i]
            length = self.doc_len[i]
            score = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (length / self.avg_len if self.avg_len else 1)
                )
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((section, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class CourseLibrary:
    """A course's notes and past assessments, loaded and searchable."""

    def __init__(self, course_dir: Path):
        self.course_dir = course_dir
        self.sections: list[Section] = []
        self.warnings: list[str] = []
        self.index: BM25Index | None = None
        self.reload()

    def reload(self) -> int:
        notes, warn_a = load_folder(self.course_dir / "notes", "notes")
        exams, warn_b = load_folder(self.course_dir / "assessments", "assessment")
        schemes, _ = load_folder(
            self.course_dir / "assessments_private", "marking_scheme"
        )
        self.sections = notes + exams + schemes
        # A missing assessments_private/ folder is normal, not a problem.
        self.warnings = warn_a + warn_b
        self.index = BM25Index(self.sections)
        return len(self.sections)

    def of_kind(self, kind: Kind) -> list[Section]:
        return [s for s in self.sections if s.kind == kind]

    def topics(self, kind: Kind | None = None) -> dict[str, list[Section]]:
        grouped: dict[str, list[Section]] = {}
        for section in self.sections:
            if kind is not None and section.kind != kind:
                continue
            grouped.setdefault(section.topic, []).append(section)
        return grouped

    def search(
        self,
        query: str,
        top_k: int = 5,
        kind: Kind | tuple[Kind, ...] | None = None,
    ) -> list[tuple[Section, float]]:
        return self.index.search(query, top_k, kind) if self.index else []

    def get(self, source: str, locator: str) -> Section | None:
        """Fetch one section. Matching is case-insensitive and forgiving."""
        source_l, locator_l = source.strip().lower(), locator.strip().lower()
        for section in self.sections:
            if section.source.lower() == source_l and section.locator.lower() == locator_l:
                return section
        for section in self.sections:
            if section.source.lower() == source_l and locator_l in section.locator.lower():
                return section
        return None
