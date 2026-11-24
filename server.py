"""Course Tutor - an MCP server over a teacher's own course materials.

Design note
-----------
The tools here only *retrieve*. They find the right passage of the lecture
notes, or the relevant past exam question, and hand it back with a citation.
The teaching itself - writing a new practice question, explaining a concept,
judging what a BSc student needs versus an MSc student - is left to the model.

That split is deliberate. Retrieval is something code does reliably and a
language model does badly, since a model asked for a page number will invent
one. Pedagogy is the reverse. Putting question-generation inside a tool would
mean writing a worse version of something the model already does well, and it
would stop the answers depending on this teacher's actual material.

The practical consequence: every answer a student gets is grounded in a real
section of the course, and cites it, so the student can go and read the source.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from notes_index import CourseLibrary

# Point this at your own course folder with:
#   export COURSE_DIR=/path/to/your/course
# It should contain notes/ and assessments/ subfolders.
COURSE_DIR = Path(os.environ.get("COURSE_DIR", Path(__file__).parent / "course"))

INSTRUCTIONS = """
This server exposes one teacher's course: lecture notes in notes/, past
exams and quizzes in assessments/, and marking schemes in
assessments_private/.

This server is the teacher's own, running on their machine, so it can see
all three. The class web app scopes retrieval by role and never lets a
student reach a marking scheme.

Ground every answer about course content in a tool result, and cite the
section you used so the student can read it. Call list_topics when you need
to know what the course covers before answering.

When asked for practice questions, do two things: search_notes for the
subject matter, and find_past_questions to see how this teacher actually
asks about it - the phrasing, the mark allocation, the level of detail
expected. Write new questions in that style, from what the notes contain.

If the notes do not cover something, say so rather than answering from
general knowledge. The student is revising for this course, not a different
one, and material that is not in the notes will not be on the exam.
"""

mcp = MCPServer(
    name="course-tutor",
    title="Course Tutor",
    instructions=INSTRUCTIONS.strip(),
    version="0.1.0",
)

library = CourseLibrary(COURSE_DIR)


def _format(section, score: float | None = None) -> dict:
    out = {
        "citation": section.citation,
        "topic": section.topic,
        "source": section.source,
        "locator": section.locator,
        "text": section.text,
    }
    if section.marks is not None:
        out["marks"] = section.marks
    if score is not None:
        out["relevance"] = round(score, 2)
    return out


@mcp.tool()
def list_topics() -> dict:
    """List what the course covers: topics, sections and past papers.

    Call this when you need to know the scope of the course, when a student
    asks what is on the syllabus, or before telling a student that something
    is not covered. Returns lecture note topics and the assessments
    available, with the file each came from.
    """
    notes = library.topics("notes")
    exams = library.topics("assessment")

    result = {
        "course_folder": str(library.course_dir),
        "lecture_topics": [
            {
                "topic": topic,
                "sources": sorted({s.source for s in sections}),
                "sections": [s.locator for s in sections],
            }
            for topic, sections in sorted(notes.items())
        ],
        "assessments": [
            {
                "topic": topic,
                "sources": sorted({s.source for s in sections}),
                "question_count": len(sections),
            }
            for topic, sections in sorted(exams.items())
        ],
        "marking_schemes": [
            {"topic": topic, "sources": sorted({s.source for s in sections})}
            for topic, sections in sorted(library.topics("marking_scheme").items())
        ],
        "section_count": len(library.sections),
    }
    if library.warnings:
        result["warnings"] = library.warnings
    return result


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> dict:
    """Search the lecture notes and return the most relevant sections.

    Use this for any question about course content, before answering from
    general knowledge. Good queries are the student's own words or the key
    terms from them, for example "why does the lag phase happen" or "gram
    stain crystal violet". Returns matching sections with their full text and
    a citation, which for a PDF includes the page number.

    Args:
        query: What to look for, in plain language or as key terms.
        top_k: How many sections to return. Default 5, maximum 20.
    """
    top_k = max(1, min(int(top_k), 20))
    hits = library.search(query, top_k, kind="notes")

    if not hits:
        return {
            "query": query,
            "result_count": 0,
            "results": [],
            "note": (
                "Nothing in the lecture notes matched. Tell the student this "
                "is not covered in the notes rather than answering from "
                "general knowledge, or call list_topics to see what is."
            ),
        }

    return {
        "query": query,
        "result_count": len(hits),
        "results": [_format(section, score) for section, score in hits],
    }


@mcp.tool()
def find_past_questions(query: str, top_k: int = 5) -> dict:
    """Find questions on a topic from this course's past exams and quizzes.

    Use this before writing practice questions, so new ones match how this
    teacher actually asks: the phrasing, the mark allocation, the depth
    expected. Also use it when a student asks what has come up before, or
    wants to practise on real past questions.

    Args:
        query: The topic to look for, e.g. "growth curve" or "gram stain".
        top_k: How many questions to return. Default 5, maximum 20.
    """
    top_k = max(1, min(int(top_k), 20))
    hits = library.search(query, top_k, kind="assessment")

    if not hits:
        available = len(library.of_kind("assessment"))
        return {
            "query": query,
            "result_count": 0,
            "results": [],
            "note": (
                f"No past question matched. There are {available} indexed. "
                "Write practice questions from the lecture notes instead, and "
                "say that no past question on this topic was found."
                if available
                else "No past papers are loaded. Add PDFs to the assessments/ "
                "folder and call reload_materials."
            ),
        }

    return {
        "query": query,
        "result_count": len(hits),
        "results": [_format(section, score) for section, score in hits],
    }


@mcp.tool()
def find_marking_schemes(query: str, top_k: int = 3) -> dict:
    """Find marking schemes and model answers for a topic. Teacher only.

    Use this when the teacher asks how something was marked, what earned
    full marks, where students lost marks, or when judging how hard a new
    question is - the scheme shows the depth expected at each mark weight.

    Do not reproduce a marking scheme for anyone who identifies as a
    student. This tool reads assessments_private/, which exists precisely
    so that material stays separate.

    Args:
        query: The topic, e.g. "growth curve" or "gram stain".
        top_k: How many to return. Default 3, maximum 10.
    """
    top_k = max(1, min(int(top_k), 10))
    hits = library.search(query, top_k, kind="marking_scheme")

    if not hits:
        total = len(library.of_kind("marking_scheme"))
        return {
            "query": query,
            "result_count": 0,
            "results": [],
            "note": (
                f"No marking scheme matched. {total} are indexed."
                if total
                else "No marking schemes are loaded. Put them in "
                "assessments_private/ and call reload_materials."
            ),
        }

    return {
        "query": query,
        "result_count": len(hits),
        "results": [_format(section, score) for section, score in hits],
    }


@mcp.tool()
def get_section(source: str, locator: str) -> dict:
    """Fetch the full text of one specific section, page or question.

    Use this when you need the surrounding detail of something a search
    returned, or when a student asks to read a section in full. Take the
    `source` and `locator` values from a search result.

    Args:
        source: The file name, e.g. "lecture_02_growth.pdf".
        locator: The section heading, page ("p. 4") or question ("Q3").
    """
    section = library.get(source, locator)
    if section is None:
        return {
            "found": False,
            "note": (
                f"No '{locator}' in '{source}'. Call list_topics to see the "
                "exact sources and locators available."
            ),
        }
    return {"found": True, **_format(section)}


@mcp.tool()
def reload_materials() -> dict:
    """Re-read the course folder after adding or editing files.

    Use this when the teacher says they have added notes or a new past
    paper, so the index picks it up without restarting the server.
    """
    count = library.reload()
    result = {
        "reloaded": True,
        "course_folder": str(library.course_dir),
        "section_count": count,
        "note_sections": len(library.of_kind("notes")),
        "past_questions": len(library.of_kind("assessment")),
        "marking_schemes": len(library.of_kind("marking_scheme")),
    }
    if library.warnings:
        result["warnings"] = library.warnings
    return result


if __name__ == "__main__":
    mcp.run()
