#!/usr/bin/env python3
"""Re-run labs and confirm they reproduce their committed results (CI).

    uv run python tools/verify_labs.py              every lab that has committed results
    uv run python tools/verify_labs.py 2.3 1.3      only these sessions
    uv run python tools/verify_labs.py --changed    labs whose files changed vs origin/main

A lab passes when it exits 0 (its own asserts hold) and every number in its results.json matches
the committed value within a relative tolerance of 1e-6 (floating-point summation order can differ
across platforms and BLAS builds; anything larger is a real change and must be committed).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import run_lab  # noqa: E402
from course import ROOT, Course  # noqa: E402

REL_TOL, ABS_TOL = 1e-6, 1e-9

# SCS (an ADMM solver) can converge to a materially different near-optimal point across
# platforms/BLAS builds than the one committed — most visibly in Session 3.4's "daily units"
# case, which is deliberately badly scaled to demonstrate solver fragility. That is the lesson,
# not a bug, so its residual-vs-reference diagnostics get a looser absolute tolerance instead of
# the strict default. Each override stays >=100x tighter than that lab's own semantic assert on
# the same quantity (dvol_bp < 0.1, max_dw < 1e-2), so a real regression is still caught.
LOOSE_ABS_TOL = {"_scs_dvol_bp": 1e-3, "_scs_dw": 1e-4, "_scs_minw": 1e-5}


def tol_for(key: str) -> tuple[float, float]:
    for suffix, atol in LOOSE_ABS_TOL.items():
        if suffix in key:
            return REL_TOL, atol
    return REL_TOL, ABS_TOL


def changed_sessions(course: Course) -> set[str]:
    out = subprocess.run(["git", "diff", "--name-only", "origin/main...HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    ids = set()
    for s in course.sessions:
        stem = s.repo_rel(s.lab_py).rsplit(".", 1)[0]
        if any(p.startswith(stem) or p == s.repo_rel(s.results_json) for p in out):
            ids.add(s.id)
    if any(p.startswith("src/quantnb/") for p in out):  # the package changed: re-verify everything
        ids = {s.id for s in course.sessions if s.lab_py.exists()}
    return ids


def compare(old: dict, new: dict) -> list[str]:
    problems = []
    for key in sorted(set(old) | set(new)):
        if key not in new:
            problems.append(f"{key}: missing after re-run")
        elif key not in old:
            problems.append(f"{key}: new result not committed")
        else:
            a, b = old[key], new[key]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                rtol, atol = tol_for(key)
                if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                    problems.append(f"{key}: committed {a!r} ≠ re-run {b!r}")
            elif a != b:
                problems.append(f"{key}: committed {a!r} ≠ re-run {b!r}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--changed", action="store_true")
    args = ap.parse_args()
    course = Course()
    if args.changed:
        wanted = changed_sessions(course)
    elif args.ids:
        wanted = set(args.ids)
    else:
        wanted = {s.id for s in course.sessions if s.lab_py.exists()}
    failed = 0
    for s in course.sessions:
        if s.id not in wanted or not s.lab_py.exists():
            continue
        committed = json.loads(s.results_json.read_text(encoding="utf-8"))["results"] if s.results_json.exists() else None
        if not run_lab(s):
            failed += 1
            continue
        if committed is None:
            print(f"  FAIL {s.id}: no committed results.json to compare against")
            failed += 1
            continue
        fresh = json.loads(s.results_json.read_text(encoding="utf-8"))["results"]
        problems = compare(committed, fresh)
        for p in problems:
            print(f"  FAIL {s.id}: {p}")
        failed += bool(problems)
        if not problems:
            print(f"  ok   {s.id}: {len(fresh)} results reproduced")
    print(f"verify_labs: {len(wanted)} lab(s), {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
