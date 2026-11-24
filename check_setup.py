"""Check your setup before running anything else.

Verifies that your course material loads, that your API key works, and
prints the model names your account can actually use - which is the usual
cause of a BadRequestError when the app otherwise looks fine.

    python check_setup.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(HERE / "webapp" / ".env")
except ImportError:
    pass

from notes_index import CourseLibrary  # noqa: E402


def check_material() -> bool:
    course_dir = Path(os.environ.get("COURSE_DIR", HERE / "course"))
    print(f"Course folder: {course_dir}")

    if not course_dir.is_dir():
        print("  MISSING - no folder there. Set COURSE_DIR or create course/.")
        return False

    library = CourseLibrary(course_dir)
    notes = library.of_kind("notes")
    exams = library.of_kind("assessment")
    print(f"  lecture sections: {len(notes)}")
    print(f"  past questions  : {len(exams)}")

    for warning in library.warnings:
        print(f"  WARNING: {warning}")

    if not notes:
        print("  No lecture notes found. Put files in course/notes/.")
        return False
    return True


def check_api() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("\nNo ANTHROPIC_API_KEY set.")
        print("  The MCP server works without one. The web app does not.")
        print("  Put it in webapp/.env to use the web app.")
        return False

    # Never print the key itself.
    print(f"\nAPI key: set ({len(key)} characters, ends ...{key[-4:]})")

    try:
        import anthropic
    except ImportError:
        print("  The anthropic package isn't installed.")
        print("  Run: pip install -r webapp/requirements.txt")
        return False

    client = anthropic.Anthropic(api_key=key)

    try:
        models = list(client.models.list(limit=20).data)
    except Exception as exc:
        print(f"  The key was rejected: {exc}")
        print("  Check it is correct and that the key has not been revoked.")
        return False

    print("  Key works. Models your account can use:\n")
    for model in models:
        print(f"    {model.id}")

    wanted = os.environ.get("MODEL", "claude-haiku-5-5")
    available = {m.id for m in models}
    print(f"\n  Configured MODEL: {wanted}")

    if wanted in available:
        print("  This model is available. You're set.")
        return True

    print("  NOT in the list above - this is what causes a BadRequestError.")
    cheap = [m for m in models if "haiku" in m.id.lower()]
    suggestion = (cheap or models)[0].id if models else None
    if suggestion:
        print(f"\n  Add this line to webapp/.env:\n\n    MODEL={suggestion}\n")
    return False


if __name__ == "__main__":
    print("=" * 60)
    ok_material = check_material()
    ok_api = check_api()
    print("=" * 60)
    if ok_material and ok_api:
        print("All good. Run the web app with:  cd webapp && python app.py")
    else:
        print("Fix the items above, then run this again.")
    sys.exit(0 if (ok_material and ok_api) else 1)
