#!/usr/bin/env python3
"""Create a session page and its companion lab from the templates (orchestrator tool).

    uv run python tools/scaffold.py session 2.3                 page + lab stub (refuses to overwrite)
    uv run python tools/scaffold.py session 2.3 --status doing  … and mark it in progress in course.yaml
    uv run python tools/scaffold.py status 2.3 published        just change the status

Metadata tokens are filled from course.yaml; content tokens ({{S1_TITLE}}, {{SUBTITLE}}, …) are
left for the author. Run tools/sync_nav.py afterwards to fill the generated regions.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course import COURSE_YAML, LEVEL_DURATION, Course, read_text, write_text  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent / "templates"
STATUSES = ("planned", "doing", "published")


def fill(text: str, tokens: dict) -> str:
    for k, v in tokens.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def set_status(sid: str, status: str) -> None:
    if status not in STATUSES:
        sys.exit(f"status must be one of {STATUSES}")
    lines = read_text(COURSE_YAML).split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(rf'\s*- id: "{re.escape(sid)}"\s*$', ln)), None)
    if start is None:
        sys.exit(f"session {sid} not found in course.yaml")
    for j in range(start + 1, len(lines)):
        if re.match(r"\s*- id: ", lines[j]):
            break
        if re.match(r"\s*status: ", lines[j]):
            lines[j] = re.sub(r"status: \S+", f"status: {status}", lines[j])
            write_text(COURSE_YAML, "\n".join(lines))
            print(f"course.yaml: {sid} → {status}")
            return
    sys.exit(f"no status line for {sid}")


def cmd_session(a) -> None:
    c = Course()
    s = c.by_id.get(a.id) or sys.exit(f"unknown session {a.id}")
    _, nxt = c.prev_next(s)
    page_tokens = {
        "SESSION_ID": s.id,
        "DURATION": f"{LEVEL_DURATION} + lab",
        "LEVEL": s.level,
        "TRACKS": " · ".join(s.tracks),
        "NEXT_ID": nxt.id if nxt else s.id,
    }
    if s.path.exists() and not a.force:
        print(f"kept existing {s.rel_path}")
    else:
        write_text(s.path, fill(read_text(TEMPLATES / "session.html"), page_tokens))
        print(f"wrote site/{s.rel_path}")
    lab_tokens = {
        "ID": s.id, "TITLE": s.title, "UNIT": s.unit_no, "SESSION": s.no, "GOAL": s.goal,
        "PAGE_URL": c.canonical(s.rel_path), "REPO": c.repo, "BRANCH": c.branch,
    }
    if s.lab_py.exists() and not a.force:
        print(f"kept existing {s.repo_rel(s.lab_py)}")
    else:
        write_text(s.lab_py, fill(read_text(TEMPLATES / "lab.py"), lab_tokens))
        print(f"wrote {s.repo_rel(s.lab_py)}")
    if a.status:
        set_status(s.id, a.status)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("session", help="create page + lab for a session id")
    p.add_argument("id")
    p.add_argument("--status", choices=STATUSES)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_session)
    q = sub.add_parser("status", help="set a session's status in course.yaml")
    q.add_argument("id")
    q.add_argument("status", choices=STATUSES)
    q.set_defaults(func=lambda a: set_status(a.id, a.status))
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
