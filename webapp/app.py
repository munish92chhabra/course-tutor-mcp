"""Class-facing web app for the course tutor.

Why this exists alongside the MCP server
----------------------------------------
The MCP server is for the teacher: it runs on your machine, over your
files, inside your own Claude. Students can't use it - a custom connector
needs its own Claude account with connector access, and the server has to be
publicly reachable. Requiring sixty students to hold subscriptions is not a
course plan.

So this is the same retrieval, served as a web page. A student opens a URL
on a phone, types a question, and gets an answer grounded in the lecture
notes with citations. No install, no account.

The retrieval is `notes_index.py`, imported unchanged. That module was
written with no MCP dependency precisely so it could be reused here.

Shape of a request
------------------
    question -> search the notes -> search past papers
             -> build a grounded prompt -> stream Claude's answer back

One retrieval, one model call. Not tool-use: for a single student question
the extra round trips buy nothing, and a predictable one-call cost is what
makes a class budget forecastable.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from anthropic import APIError, AsyncAnthropic
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.middleware.sessions import SessionMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from notes_index import (  # noqa: E402
    STUDENT_KINDS,
    TEACHER_KINDS,
    CourseLibrary,
    Section,
)

# Load settings from a .env file sitting next to this one, if there is one,
# so you don't have to retype your API key on every run. Values already set
# in the real environment win, which is what you want on a deployed host.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # python-dotenv not installed: environment only
    pass

# --------------------------------------------------------------------------
# Configuration - everything secret comes from the environment
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
COURSE_DIR = Path(os.environ.get("COURSE_DIR", HERE.parent / "course"))

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Two passwords, two roles. The password decides which you get: students
# see notes and past questions, teachers additionally see marking schemes.
CLASS_PASSWORD = os.environ.get("CLASS_PASSWORD", "")
TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "")

SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)

MODEL = os.environ.get("MODEL", "claude-haiku-5-5")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2000"))

#: How many lecture sections and past questions to put in front of the model.
NOTE_HITS = int(os.environ.get("NOTE_HITS", "4"))
EXAM_HITS = int(os.environ.get("EXAM_HITS", "2"))

#: How many marking-scheme extracts a teacher sees per question.
SCHEME_HITS = int(os.environ.get("SCHEME_HITS", "2"))

#: Rate limit, per browser session.
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "20"))  # questions
RATE_WINDOW = int(os.environ.get("RATE_WINDOW", "3600"))  # seconds

#: Hard ceiling on one question, to stop someone pasting an essay in.
MAX_QUESTION_CHARS = 600

#: Hard ceiling on retrieved context, so one enormous page cannot blow up
#: the per-question cost.
MAX_CONTEXT_CHARS = 14_000

COURSE_NAME = os.environ.get("COURSE_NAME", "Course")

#: Where questions are logged, so the teacher can see what the class asked.
#: Set QUESTION_LOG="" to turn logging off entirely.
QUESTION_LOG = os.environ.get("QUESTION_LOG", str(HERE / "question_log.jsonl"))


STUDENT_PROMPT = """
You are a teaching assistant for a university {course} course. You are
talking to a student revising for their exams.

You will be given extracts from their teacher's own notes, and sometimes
questions from past papers for this course. Answer only from those extracts.

Rules you must follow:

- Ground every factual claim in the extracts, and cite the source in
  brackets after it, exactly as given, for example (lecture_02_growth.pdf,
  p. 4). The student needs to be able to find it and read it.
- If the extracts do not answer the question, say so plainly: "That isn't
  covered in the notes I have." Then say what the notes do cover nearby.
  Never fill the gap from general knowledge - material that is not in these
  notes will not be on their exam, and a confident off-syllabus answer is
  worse than no answer.
- Explain, don't just quote. Put it in your own words at the level of an
  undergraduate who has attended the lectures, then cite.
- Be concise. A few short paragraphs. Use a list only when the content is
  genuinely a list.

Answering at the right length for the marks:

When a past question is included and carries a mark value, tell the student
what an answer at that weight needs - how many distinct points, and how much
depth each one needs. Work it out from the notes. As a rough guide:

- 2 marks: two crisp facts, or one fact plus its consequence. A sentence or
  two. No preamble.
- 4 marks: a short structured explanation, usually three or four linked
  points, each stated and briefly justified.
- 6 marks: several linked points with reasoning, often split across two
  halves of the question - say, structure and function - where both halves
  are needed for full marks.

Say which specific points from the notes you would include, and roughly how
long the answer should be. This is the thing students most often get wrong:
they write a paragraph for 2 marks and a sentence for 6.

You do not have the teacher's marking scheme and must not pretend otherwise.
Frame this as your guidance from the notes, not as how the paper is marked.
If a student asks for the official answer or mark scheme, tell them you
don't have it, and offer to work through the question with them instead.

You are not marking anything. If a student asks you to grade their answer,
help them check it against the notes instead.
""".strip()


TEACHER_PROMPT = """
You are an assistant to the teacher of a university {course} course. You are
talking to the teacher, not a student.

You will be given extracts from their lecture notes, their past papers, and
where relevant their marking schemes. The marking schemes are confidential
and you may discuss them freely here, because this session is the teacher's
own.

Rules you must follow:

- Ground every factual claim in the extracts and cite the source in brackets
  exactly as given, for example (lecture_02_growth.pdf, p. 4).
- If the material does not cover something, say so rather than inventing it.
- Be concise and direct. This is a colleague, not a student.

When a marking scheme is included, use it. Say what earns marks and where
the scheme caps an answer, and quote the mark split when it is relevant.

Rating difficulty:

When you propose a question, or the teacher shows you one, rate it easy,
medium or hard, and justify the rating from the material rather than from
general intuition. Weigh:

- Coverage. Is the answer stated plainly in the notes, or must it be
  assembled from several places? Thin coverage makes a question harder, and
  is worth flagging separately as a possible gap in the notes.
- Precedent. Has something similar been asked in a past paper? If so,
  students will have seen it, which makes it easier regardless of content.
- Steps. How many linked inferences are needed? One recalled fact is easy;
  a calculation plus interpretation is medium; applying a principle to an
  unfamiliar scenario is hard.

Give the rating, one line of justification, and a suggested mark allocation.
If a question is harder than its mark weight suggests, say so - that
mismatch is the most useful thing you can flag.

Do this even when the teacher asked a content question rather than asking
for a new question: if past questions or a marking scheme were retrieved,
close with one short line on how the topic has been examined before, at what
difficulty, and where students lost marks. Keep it to a line or two. If
nothing relevant was retrieved, say nothing rather than padding.
""".strip()


# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(title=f"{COURSE_NAME} Tutor")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, https_only=False)

library = CourseLibrary(COURSE_DIR)
client = AsyncAnthropic(api_key=API_KEY) if API_KEY else None

#: session id -> timestamps of recent questions. In memory, so it resets on
#: restart and is per-process. Fine for one class on one small instance; if
#: you ever run more than one worker, move this to Redis.
_history: dict[str, deque[float]] = {}


@dataclass
class Retrieved:
    notes: list[tuple[Section, float]]
    exams: list[tuple[Section, float]]
    schemes: list[tuple[Section, float]]  # always empty for a student

    @property
    def is_empty(self) -> bool:
        return not (self.notes or self.exams or self.schemes)

    def sources(self) -> list[dict]:
        seen: list[dict] = []
        for section, _ in self.notes + self.exams + self.schemes:
            entry = {
                "citation": section.citation,
                "topic": section.topic,
                "kind": section.kind,
            }
            if entry not in seen:
                seen.append(entry)
        return seen


def retrieve(question: str, role: str) -> Retrieved:
    """Search within what this role is allowed to see.

    A student session never searches marking schemes at all. This is the
    enforcement point: it is not left to the model to decline, because a
    model that has the text in its context has already been given it.
    """
    schemes: list[tuple[Section, float]] = []
    if role == "teacher":
        schemes = library.search(question, SCHEME_HITS, kind="marking_scheme")

    return Retrieved(
        notes=library.search(question, NOTE_HITS, kind="notes"),
        exams=library.search(question, EXAM_HITS, kind="assessment"),
        schemes=schemes,
    )


def role_of(request: Request) -> str:
    """The role this session signed in as. Defaults to the safer one."""
    role = request.session.get("role", "student")
    if role == "teacher" and request.session.get("view_as_student"):
        return "student"  # teacher previewing what the class sees
    return role if role in ("student", "teacher") else "student"


def build_context(found: Retrieved) -> str:
    """Lay the extracts out for the model, with citations attached."""
    parts: list[str] = []
    budget = MAX_CONTEXT_CHARS

    if found.notes:
        parts.append("=== EXTRACTS FROM THE LECTURE NOTES ===")
        for section, _ in found.notes:
            block = f"\n[{section.citation}]\n{section.text}"
            if len(block) > budget:
                block = block[:budget] + "\n[...extract truncated...]"
            parts.append(block)
            budget -= len(block)
            if budget <= 0:
                break

    if found.exams and budget > 0:
        parts.append("\n=== RELATED QUESTIONS FROM PAST PAPERS ===")
        for section, _ in found.exams:
            marks = f" [{section.marks} marks]" if section.marks is not None else ""
            block = f"\n[{section.citation}{marks}]\n{section.text}"
            if len(block) > budget:
                block = block[:budget] + "\n[...truncated...]"
            parts.append(block)
            budget -= len(block)
            if budget <= 0:
                break

    if found.schemes and budget > 0:
        parts.append("\n=== MARKING SCHEMES (CONFIDENTIAL - TEACHER ONLY) ===")
        for section, _ in found.schemes:
            block = f"\n[{section.citation}]\n{section.text}"
            if len(block) > budget:
                block = block[:budget] + "\n[...truncated...]"
            parts.append(block)
            budget -= len(block)
            if budget <= 0:
                break

    return "\n".join(parts)


def rate_limited(session_id: str) -> int | None:
    """Return seconds to wait if over the limit, else None."""
    now = time.time()
    seen = _history.setdefault(session_id, deque())
    while seen and now - seen[0] > RATE_WINDOW:
        seen.popleft()
    if len(seen) >= RATE_LIMIT:
        return int(RATE_WINDOW - (now - seen[0])) + 1
    seen.append(now)
    return None


def log_question(
    session_id: str,
    role: str,
    question: str,
    found: "Retrieved",
) -> None:
    """Append one question to the log, for the teacher to review later.

    What this records and why: the question, whether the notes answered it,
    and which sections were used. A question that retrieved nothing is the
    most useful line in the file - it is either a gap in the notes or a gap
    in how they are worded.

    What it does not record: any identity. There are no accounts, so the
    session id is a random per-browser token and nothing links it to a
    person. It is kept only so you can tell one student asking five
    follow-ups from five students asking once.

    Tell your class the questions are logged. It costs nothing and it is
    the difference between analytics and surveillance.
    """
    if not QUESTION_LOG:
        return

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": session_id[:8],
        "role": role,
        "question": question,
        "answered": not found.is_empty,
        "notes_used": [s.citation for s, _ in found.notes],
        "past_questions_used": [s.citation for s, _ in found.exams],
    }

    try:
        with open(QUESTION_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:  # never let logging break a student's question
        print(f"[log] could not write question log: {exc}", file=sys.stderr)


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/config")
async def config(request: Request) -> JSONResponse:
    """What the page needs to render before anyone has asked anything."""
    return JSONResponse(
        {
            "course_name": COURSE_NAME,
            "authenticated": bool(request.session.get("ok")),
            "password_required": bool(CLASS_PASSWORD or TEACHER_PASSWORD),
            "ready": client is not None,
            "role": request.session.get("role", "student"),
            "viewing_as_student": bool(request.session.get("view_as_student")),
            "topics": sorted({s.topic for s in library.of_kind("notes")}),
            "questions_per_hour": RATE_LIMIT,
        }
    )


@app.post("/api/login")
async def login(request: Request) -> JSONResponse:
    body = await request.json()
    supplied = str(body.get("password", ""))

    if not (CLASS_PASSWORD or TEACHER_PASSWORD):
        request.session.update(ok=True, role="student", sid=secrets.token_hex(8))
        return JSONResponse({"ok": True, "role": "student"})

    # Check the teacher password first, and compare both in constant time so
    # the response time does not leak which one was closer.
    is_teacher = bool(TEACHER_PASSWORD) and secrets.compare_digest(
        supplied, TEACHER_PASSWORD
    )
    is_student = bool(CLASS_PASSWORD) and secrets.compare_digest(
        supplied, CLASS_PASSWORD
    )

    if is_teacher or is_student:
        role = "teacher" if is_teacher else "student"
        request.session.update(
            ok=True, role=role, sid=secrets.token_hex(8), view_as_student=False
        )
        return JSONResponse({"ok": True, "role": role})

    await asyncio.sleep(0.5)  # blunt the edge off guessing
    return JSONResponse({"ok": False, "error": "That password isn't right."}, 401)


@app.post("/api/view_as")
async def view_as(request: Request) -> JSONResponse:
    """Let a teacher preview exactly what the class sees, and switch back.

    Only a teacher session can do this, and it only ever reduces access.
    """
    if request.session.get("role") != "teacher":
        return JSONResponse({"error": "Not available."}, 403)

    body = await request.json()
    request.session["view_as_student"] = bool(body.get("as_student"))
    return JSONResponse(
        {"ok": True, "viewing_as_student": request.session["view_as_student"]}
    )


@app.post("/api/ask")
async def ask(request: Request):
    if (CLASS_PASSWORD or TEACHER_PASSWORD) and not request.session.get("ok"):
        return JSONResponse({"error": "Not signed in."}, 401)

    role = role_of(request)

    if client is None:
        return JSONResponse(
            {"error": "The server has no API key set. Ask your teacher."}, 503
        )

    body = await request.json()
    question = str(body.get("question", "")).strip()

    if not question:
        return JSONResponse({"error": "Type a question first."}, 400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {"error": f"Please keep questions under {MAX_QUESTION_CHARS} characters."},
            400,
        )

    session_id = request.session.setdefault("sid", secrets.token_hex(8))
    wait = rate_limited(session_id)
    if wait is not None:
        minutes = max(1, wait // 60)
        return JSONResponse(
            {
                "error": f"That's {RATE_LIMIT} questions this hour, which is the "
                f"limit. Try again in about {minutes} minute(s)."
            },
            429,
        )

    found = retrieve(question, role)
    log_question(session_id, role, question, found)

    async def stream():
        truncated = False

        if found.is_empty:
            yield sse(
                "text",
                {
                    "text": "I couldn't find anything in the lecture notes on that. "
                    "Try different words, or check with your teacher whether it's "
                    "part of the course."
                },
            )
            yield sse("done", {"sources": [], "truncated": False})
            return

        asker = "TEACHER" if role == "teacher" else "STUDENT"
        prompt = (
            f"{build_context(found)}\n\n"
            f"=== THE {asker}'S QUESTION ===\n{question}"
        )
        system = TEACHER_PROMPT if role == "teacher" else STUDENT_PROMPT

        try:
            async with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system.format(course=COURSE_NAME),
                messages=[{"role": "user", "content": prompt}],
            ) as response:
                async for chunk in response.text_stream:
                    yield sse("text", {"text": chunk})

                # Did it finish, or run out of room? A silently truncated
                # answer reads like a bug in the notes rather than a limit.
                try:
                    final = await response.get_final_message()
                    truncated = final.stop_reason == "max_tokens"
                except Exception:
                    truncated = False
        except APIError as exc:
            # Surface what the API actually said. A bare exception class name
            # ("BadRequestError") tells nobody anything; the message usually
            # names the problem outright, e.g. an invalid model name.
            detail = getattr(exc, "message", None) or str(exc)
            print(f"[ask] API error: {detail}", file=sys.stderr, flush=True)
            yield sse("error", {"error": f"The model refused the request: {detail}"})
            return
        except Exception:
            yield sse("error", {"error": "Something went wrong answering that."})
            return

        yield sse(
            "done",
            {
                "sources": found.sources(),
                "role": role,
                "truncated": truncated,
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "sections": len(library.sections),
            "notes": len(library.of_kind("notes")),
            "past_questions": len(library.of_kind("assessment")),
            "marking_schemes": len(library.of_kind("marking_scheme")),
            "model": MODEL,
            "api_key_set": client is not None,
            "class_password_set": bool(CLASS_PASSWORD),
            "teacher_password_set": bool(TEACHER_PASSWORD),
            "question_log": QUESTION_LOG or "disabled",
            "warnings": library.warnings,
        }
    )


if __name__ == "__main__":
    import uvicorn

    if not CLASS_PASSWORD and not TEACHER_PASSWORD:
        print("WARNING: no password set - anyone with the URL can use this,")
        print("         and every question is billed to your API key.")
    if library.of_kind("marking_scheme") and not TEACHER_PASSWORD:
        print("NOTE:    marking schemes are loaded but TEACHER_PASSWORD is not")
        print("         set, so nobody can reach them through the web app.")
    if not API_KEY:
        print("WARNING: ANTHROPIC_API_KEY is not set - questions will be refused.")

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
