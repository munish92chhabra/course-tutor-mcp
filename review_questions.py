"""Read the question log and tell the teacher what the class asked.

    python review_questions.py                  # everything
    python review_questions.py --since today    # just today's session
    python review_questions.py --all            # include every question verbatim

The section that matters most is UNANSWERED. Every line there is a question
your notes did not cover - either a genuine gap, or a topic your students
describe in words your notes never use. Both are worth knowing before the
next lecture.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = Path(os.environ.get("QUESTION_LOG", HERE / "webapp" / "question_log.jsonl"))

# Words too common to be worth counting as a topic.
STOP = {
    "a", "about", "an", "and", "answer", "are", "as", "ask", "at", "be",
    "between", "but", "by", "can", "could", "describe", "did", "do", "does",
    "explain", "for", "from", "give", "happen", "has", "have", "how", "i",
    "if", "in", "is", "it", "its", "just", "know", "many", "me", "mean",
    "means", "much", "my", "need", "not", "of", "on", "or", "please", "question",
    "say", "should", "so", "some", "tell", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "up", "us", "use",
    "was", "we", "what", "when", "where", "which", "why", "will", "with",
    "would", "you", "your",
}

WORD = re.compile(r"[a-z][a-z0-9-]{2,}")


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"No log at {path}")
        print("Questions are written there once students start asking.")
        sys.exit(1)

    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  (skipped malformed line {line_no})", file=sys.stderr)
    return rows


def filter_since(rows: list[dict], since: str | None) -> list[dict]:
    if not since:
        return rows

    now = datetime.now(timezone.utc)
    if since == "today":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif since == "week":
        cutoff = now - timedelta(days=7)
    else:
        try:
            cutoff = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"Can't read --since {since!r}. Use 'today', 'week', or 2026-11-25.")
            sys.exit(1)

    out = []
    for row in rows:
        try:
            if datetime.fromisoformat(row["ts"]) >= cutoff:
                out.append(row)
        except (KeyError, ValueError):
            continue
    return out


def heading(text: str) -> None:
    print(f"\n{text}")
    print("-" * len(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--since", help="today, week, or a date like 2026-11-25")
    parser.add_argument("--all", action="store_true", help="print every question")
    args = parser.parse_args()

    rows = filter_since(load(args.log), args.since)
    students = [r for r in rows if r.get("role") != "teacher"]

    if not students:
        print("No student questions in that range.")
        return

    sessions = defaultdict(list)
    for row in students:
        sessions[row.get("session", "?")].append(row)

    unanswered = [r for r in students if not r.get("answered")]

    # --- the shape of the session ---
    heading("OVERVIEW")
    print(f"  questions      : {len(students)}")
    print(f"  students (approx): {len(sessions)}  (distinct browsers)")
    per = len(students) / len(sessions)
    print(f"  questions each : {per:.1f} average")

    repeat = sum(1 for qs in sessions.values() if len(qs) > 1)
    print(f"  came back      : {repeat} of {len(sessions)} asked more than once")

    answered_pct = 100 * (len(students) - len(unanswered)) / len(students)
    print(f"  answered       : {answered_pct:.0f}% found something in the notes")

    # --- the valuable bit ---
    heading(f"UNANSWERED - {len(unanswered)} question(s) your notes did not cover")
    if not unanswered:
        print("  None. Everything asked was covered.")
    else:
        # Group repeats: three students asking the same thing is a far
        # stronger signal than three separate lines suggest.
        grouped = Counter(r["question"].strip().lower() for r in unanswered)
        askers = defaultdict(set)
        for row in unanswered:
            askers[row["question"].strip().lower()].add(row.get("session"))

        for question, times in grouped.most_common():
            people = len(askers[question])
            if times > 1:
                print(f"  - {question}   ({times}x, {people} student(s))")
            else:
                print(f"  - {question}")

        print("\n  Each of these is either a gap in the notes, or a topic your")
        print("  students name differently than your notes do. Worth checking which.")
        print("  Anything asked by several students is the one to fix first.")

    # --- what they were revising ---
    heading("MOST USED SECTIONS")
    cited = Counter()
    for row in students:
        for citation in row.get("notes_used", []):
            cited[citation] += 1
    if cited:
        for citation, n in cited.most_common(10):
            print(f"  {n:3d}  {citation}")
    else:
        print("  Nothing retrieved.")

    heading("MOST COMMON TERMS")
    terms = Counter()
    for row in students:
        for word in WORD.findall(row["question"].lower()):
            if word not in STOP:
                terms[word] += 1
    for word, n in terms.most_common(15):
        print(f"  {n:3d}  {word}")

    # --- repeated askers ---
    heading("FOLLOW-UP THREADS")
    threads = [(sid, qs) for sid, qs in sessions.items() if len(qs) >= 3]
    if not threads:
        print("  Nobody asked three or more questions in one sitting.")
    else:
        print(f"  {len(threads)} student(s) asked 3+ questions. Often a sign of")
        print("  something that didn't land the first time.\n")
        for sid, qs in sorted(threads, key=lambda p: -len(p[1]))[:5]:
            print(f"  session {sid} ({len(qs)} questions):")
            for row in qs:
                mark = " " if row.get("answered") else "*"
                print(f"    {mark} {row['question'][:90]}")
            print()
        print("  * = nothing found in the notes")

    heading("WHEN")
    hours = Counter()
    for row in students:
        try:
            hours[datetime.fromisoformat(row["ts"]).hour] += 1
        except (KeyError, ValueError):
            continue
    for hour in sorted(hours):
        bar = "#" * min(hours[hour], 40)
        print(f"  {hour:02d}:00 UTC  {bar} {hours[hour]}")

    if args.all:
        heading("EVERY QUESTION")
        for row in students:
            mark = " " if row.get("answered") else "*"
            print(f"  {row['ts'][11:16]} {mark} {row['question']}")

    print()


if __name__ == "__main__":
    main()
