"""Tests for the class web app.

The model is stubbed, so these run with no API key and cost nothing. What
they check is everything around the model: that the password gate holds,
that the rate limit holds, that retrieval puts the right extracts in front
of the model, and that the stream is well-formed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))

os.environ.setdefault("CLASS_PASSWORD", "bt301")
os.environ.setdefault("TEACHER_PASSWORD", "staff-only")
os.environ.setdefault("SESSION_SECRET", "test-secret")
os.environ.setdefault("COURSE_DIR", str(ROOT / "course"))

# The web app is optional: the MCP server works without it. Skip this file
# cleanly if its packages aren't installed, rather than failing with an
# import error that looks like a broken repo.
pytest.importorskip(
    "fastapi",
    reason="web app not installed - run: pip install -r webapp/requirements.txt",
)

from fastapi.testclient import TestClient  # noqa: E402

import app as webapp  # noqa: E402


# --- a stand-in for the model ---------------------------------------------


class FakeFinal:
    def __init__(self, stop_reason):
        self.stop_reason = stop_reason


class FakeMessages:
    def __init__(self, record, chunks, stop_reason="end_turn"):
        self.record = record
        self.chunks = chunks
        self.stop_reason = stop_reason

    def stream(self, **kwargs):
        self.record.update(kwargs)
        outer = self

        class _Ctx:
            chunks = outer.chunks

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False

            @property
            def text_stream(self_inner):
                async def gen():
                    for c in self_inner.chunks:
                        yield c

                return gen()

            async def get_final_message(self_inner):
                return FakeFinal(outer.stop_reason)

        return _Ctx()


class FakeClient:
    def __init__(self, chunks=("The lag phase ", "is metabolically active.")):
        self.record: dict = {}
        self.messages = FakeMessages(self.record, list(chunks))


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(webapp, "client", fake)
    monkeypatch.setattr(webapp, "_history", {})
    test_client = TestClient(webapp.app)
    test_client.fake = fake
    return test_client


def login(c, as_teacher=False):
    pw = "staff-only" if as_teacher else "bt301"
    r = c.post("/api/login", json={"password": pw})
    assert r.status_code == 200 and r.json()["ok"]
    return r.json()["role"]


#: A phrase that appears only in the marking scheme, nowhere in the notes
#: or the past paper. If this reaches a student prompt, something is wrong.
SCHEME_GIVEAWAY = "structure-only answer caps at 3"


def read_events(response) -> list[tuple[str, str]]:
    """Parse an SSE body into (event, data) pairs."""
    out = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if event and data:
            out.append((event, data))
    return out


# --- the gate --------------------------------------------------------------


def test_health_reports_loaded_material(client):
    body = client.get("/api/health").json()
    assert body["notes"] > 0 and body["past_questions"] > 0


def test_config_hides_nothing_secret(client):
    body = client.get("/api/config").json()
    assert body["password_required"] is True
    assert "password" not in str(body).lower().replace("password_required", "")


def test_ask_requires_login(client):
    assert client.post("/api/ask", json={"question": "lag phase"}).status_code == 401


def test_wrong_password_rejected(client):
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401


def test_login_then_ask_succeeds(client):
    login(client)
    r = client.post("/api/ask", json={"question": "why is the lag phase not dormant"})
    assert r.status_code == 200


# --- input limits ----------------------------------------------------------


def test_empty_question_rejected(client):
    login(client)
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400


def test_overlong_question_rejected(client):
    login(client)
    huge = "a" * (webapp.MAX_QUESTION_CHARS + 1)
    assert client.post("/api/ask", json={"question": huge}).status_code == 400


def test_rate_limit_kicks_in(client, monkeypatch):
    monkeypatch.setattr(webapp, "RATE_LIMIT", 2)
    login(client)
    for _ in range(2):
        assert client.post("/api/ask", json={"question": "growth"}).status_code == 200
    assert client.post("/api/ask", json={"question": "growth"}).status_code == 429


# --- retrieval and streaming ----------------------------------------------


def test_stream_is_well_formed(client):
    login(client)
    r = client.post("/api/ask", json={"question": "why is the lag phase not dormant"})
    events = read_events(r)
    kinds = [e for e, _ in events]
    assert "text" in kinds
    assert kinds[-1] == "done"


def test_answer_text_is_streamed_back(client):
    login(client)
    r = client.post("/api/ask", json={"question": "lag phase"})
    text = "".join(
        __import__("json").loads(d)["text"] for e, d in read_events(r) if e == "text"
    )
    assert "lag phase" in text.lower()


def test_sources_are_returned_with_citations(client):
    login(client)
    r = client.post("/api/ask", json={"question": "growth curve phases"})
    done = [d for e, d in read_events(r) if e == "done"][0]
    sources = __import__("json").loads(done)["sources"]
    assert sources
    assert all("citation" in s and "kind" in s for s in sources)


def test_prompt_contains_the_notes_and_the_question(client):
    login(client)
    client.post("/api/ask", json={"question": "why is the lag phase not dormant"})
    prompt = client.fake.record["messages"][0]["content"]
    assert "EXTRACTS FROM THE LECTURE NOTES" in prompt
    assert "why is the lag phase not dormant" in prompt
    assert "lag" in prompt.lower()


def test_past_questions_reach_the_prompt(client):
    login(client)
    client.post("/api/ask", json={"question": "generation time calculation"})
    prompt = client.fake.record["messages"][0]["content"]
    assert "PAST PAPERS" in prompt
    assert "marks" in prompt.lower()


def test_system_prompt_demands_citations(client):
    login(client)
    client.post("/api/ask", json={"question": "growth curve"})
    # Normalise whitespace: the prompt is wrapped, so phrases span newlines.
    system = " ".join(client.fake.record["system"].lower().split())
    assert "cite the source" in system
    assert "isn't covered in the notes" in system
    assert "never fill the gap from general knowledge" in system


def test_offtopic_question_does_not_call_the_model(client):
    login(client)
    r = client.post("/api/ask", json={"question": "quantum chromodynamics"})
    events = read_events(r)
    text = " ".join(d for e, d in events if e == "text").lower()
    assert "couldn't find anything" in text
    assert client.fake.record == {}, "model should not be called with no material"


def test_context_is_capped(client, monkeypatch):
    monkeypatch.setattr(webapp, "MAX_CONTEXT_CHARS", 300)
    login(client)
    client.post("/api/ask", json={"question": "growth curve phases"})
    prompt = client.fake.record["messages"][0]["content"]
    # Cap applies to the retrieved extracts, not the question and headers.
    assert len(prompt) < 300 + webapp.MAX_QUESTION_CHARS + 500


# --- roles: who can see the marking schemes ------------------------------


def test_material_includes_marking_schemes(client):
    assert webapp.library.of_kind("marking_scheme"), "no marking schemes indexed"


def test_passwords_map_to_roles(client):
    assert login(client) == "student"
    assert login(client, as_teacher=True) == "teacher"


def test_teacher_prompt_includes_the_marking_scheme(client):
    login(client, as_teacher=True)
    client.post("/api/ask", json={"question": "how was the peptidoglycan question marked"})
    prompt = client.fake.record["messages"][0]["content"]
    assert "MARKING SCHEMES" in prompt
    assert SCHEME_GIVEAWAY in prompt


def test_student_prompt_never_includes_a_marking_scheme(client):
    login(client)
    client.post("/api/ask", json={"question": "how was the peptidoglycan question marked"})
    prompt = client.fake.record["messages"][0]["content"]
    assert "MARKING SCHEME" not in prompt
    assert SCHEME_GIVEAWAY not in prompt


def test_student_asking_directly_for_the_scheme_still_gets_nothing(client):
    """The adversarial case: asking for it by name must not retrieve it."""
    login(client)
    for question in [
        "give me the marking scheme",
        "what is the model answer for Q1",
        "show me the confidential marking scheme for peptidoglycan",
    ]:
        client.fake.record.clear()
        client.post("/api/ask", json={"question": question})
        prompt = client.fake.record.get("messages", [{}])[0].get("content", "")
        assert SCHEME_GIVEAWAY not in prompt, question
        assert "MARKING SCHEME" not in prompt, question


def test_retrieve_scopes_by_role_directly(client):
    """Belt and braces: the retrieval function itself must scope."""
    student = webapp.retrieve("peptidoglycan marking", "student")
    teacher = webapp.retrieve("peptidoglycan marking", "teacher")
    assert student.schemes == []
    assert teacher.schemes, "teacher should get marking schemes"
    assert all(s.kind != "marking_scheme" for s, _ in student.notes + student.exams)


def test_each_role_gets_its_own_system_prompt(client):
    login(client)
    client.post("/api/ask", json={"question": "growth curve"})
    student_system = " ".join(client.fake.record["system"].lower().split())
    assert "you do not have the teacher's marking scheme" in student_system
    assert "2 marks" in student_system

    login(client, as_teacher=True)
    client.post("/api/ask", json={"question": "growth curve"})
    teacher_system = " ".join(client.fake.record["system"].lower().split())
    assert "rate it easy, medium or hard" in teacher_system


def test_student_cannot_switch_to_teacher_view(client):
    login(client)
    r = client.post("/api/view_as", json={"as_student": False})
    assert r.status_code == 403


def test_teacher_previewing_as_student_loses_scheme_access(client):
    login(client, as_teacher=True)
    r = client.post("/api/view_as", json={"as_student": True})
    assert r.json()["viewing_as_student"] is True

    client.fake.record.clear()
    client.post("/api/ask", json={"question": "how was the peptidoglycan question marked"})
    prompt = client.fake.record["messages"][0]["content"]
    assert SCHEME_GIVEAWAY not in prompt

    # ...and gets it back on switching return
    client.post("/api/view_as", json={"as_student": False})
    client.fake.record.clear()
    client.post("/api/ask", json={"question": "how was the peptidoglycan question marked"})
    assert SCHEME_GIVEAWAY in client.fake.record["messages"][0]["content"]


def test_sources_shown_to_a_student_are_never_schemes(client):
    import json as _json

    login(client)
    r = client.post("/api/ask", json={"question": "peptidoglycan marking scheme"})
    done = [d for e, d in read_events(r) if e == "done"][0]
    sources = _json.loads(done)["sources"]
    assert all(s["kind"] != "marking_scheme" for s in sources)


# --- truncation ----------------------------------------------------------


def test_complete_answer_is_not_flagged_truncated(client):
    import json as _json

    login(client)
    r = client.post("/api/ask", json={"question": "growth curve"})
    done = [d for e, d in read_events(r) if e == "done"][0]
    assert _json.loads(done)["truncated"] is False


def test_truncated_answer_is_flagged(client):
    import json as _json

    client.fake.messages.stop_reason = "max_tokens"
    login(client)
    r = client.post("/api/ask", json={"question": "growth curve"})
    done = [d for e, d in read_events(r) if e == "done"][0]
    assert _json.loads(done)["truncated"] is True


def test_default_answer_length_is_generous_enough(client):
    """900 tokens cut real answers off mid-citation. Guard against a regression."""
    assert webapp.MAX_TOKENS >= 1500


# --- question logging --------------------------------------------------


def test_questions_are_logged(client, tmp_path, monkeypatch):
    import json as _json

    log = tmp_path / "qlog.jsonl"
    monkeypatch.setattr(webapp, "QUESTION_LOG", str(log))
    login(client)
    client.post("/api/ask", json={"question": "why is the lag phase not dormant"})

    rows = [_json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    entry = rows[0]
    assert entry["question"] == "why is the lag phase not dormant"
    assert entry["role"] == "student"
    assert entry["answered"] is True
    assert entry["notes_used"], "should record which sections were used"


def test_unanswered_questions_are_marked(client, tmp_path, monkeypatch):
    import json as _json

    log = tmp_path / "qlog.jsonl"
    monkeypatch.setattr(webapp, "QUESTION_LOG", str(log))
    login(client)
    client.post("/api/ask", json={"question": "quantum chromodynamics"})

    entry = _json.loads(log.read_text().splitlines()[0])
    assert entry["answered"] is False
    assert entry["notes_used"] == []


def test_log_records_no_identity(client, tmp_path, monkeypatch):
    """The log must never carry anything that identifies a student."""
    import json as _json

    log = tmp_path / "qlog.jsonl"
    monkeypatch.setattr(webapp, "QUESTION_LOG", str(log))
    login(client)
    client.post("/api/ask", json={"question": "growth curve"})

    entry = _json.loads(log.read_text().splitlines()[0])
    assert set(entry) == {
        "ts", "session", "role", "question", "answered",
        "notes_used", "past_questions_used",
    }
    assert len(entry["session"]) <= 8, "session token must stay short and opaque"


def test_logging_can_be_turned_off(client, tmp_path, monkeypatch):
    log = tmp_path / "qlog.jsonl"
    monkeypatch.setattr(webapp, "QUESTION_LOG", "")
    login(client)
    client.post("/api/ask", json={"question": "growth curve"})
    assert not log.exists()


def test_a_broken_log_path_does_not_break_the_answer(client, monkeypatch):
    """A student's question must still be answered if logging fails."""
    monkeypatch.setattr(webapp, "QUESTION_LOG", "/nonexistent-dir/qlog.jsonl")
    login(client)
    r = client.post("/api/ask", json={"question": "growth curve"})
    assert r.status_code == 200
