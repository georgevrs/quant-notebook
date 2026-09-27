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

# SCS (an ADMM solver) converges to a genuinely different near-optimal point from one CI run to
# the next, even on identical hardware and the same commit — observed swinging between -6e-5 and
# +1e-3 across two back-to-back Linux runs of Session 3.4's "daily units" case, which is
# deliberately badly scaled to demonstrate solver fragility. That non-reproducibility IS the
# lesson, not a bug: no fixed tolerance is the right tool for it, so these diagnostics are
# excluded from the committed-value check entirely. The lab's own live assert on the same
# quantity (dvol_bp < 0.1, max_dw < 1e-2) still runs on every verify and is what actually guards
# against a real regression.
NONDETERMINISTIC_KEYS = (
    "_scs_dvol_bp", "_scs_dw", "_scs_minw",
    # Session 4.5's three write-timing keys (seconds, no "_ms" suffix): see the note below.
    "t_write_csv", "t_write_mono_parquet", "t_write_partitioned_parquet",
    # A CSV of float columns' exact byte count is sensitive to the last digit of each float's text
    # representation, which inherits the same cross-platform/BLAS-level numeric noise that REL_TOL
    # already tolerates in the underlying data -- just amplified, because a one-bit-different float
    # can print with a different number of characters. Confirmed on Sessions 4.1/4.5/4.6: fixing the
    # dominant cause (to_csv defaulted to the platform line ending, CRLF on Windows vs LF on Linux --
    # now pinned to "\n" in all three labs) took the gap from ~1.4% to ~0.001%, but didn't zero it.
    # Exact key names, unique to these three sessions (checked against every other lab's keys).
    "compression_ratio", "csv_bytes", "csv_mb",
)

# Wall-clock timing benchmarks (Session 4.5's storage-stack comparison: naive CSV vs Parquet vs
# DuckDB vs polars) vary with disk cache state and machine load, especially on a shared CI runner —
# by design, not by bug. Match only the deliberate "_ms" duration suffix and the "speedup_"/
# "slowdown_" ratio prefix (checked against every other lab's keys — some legitimately use a bare
# "t_" prefix for the Student-t distribution or a period count, which must stay strictly checked).
# The lab's own directional asserts (e.g. `t_duckdb < t_naive_csv / 5`) still run on every verify.
NONDETERMINISTIC_SUFFIXES = ("_ms",)
NONDETERMINISTIC_PREFIXES = ("speedup_", "slowdown_")

# GARCH/GJR-GARCH maximum-likelihood fits (via the `arch` package's scipy-based optimizer) are more
# BLAS-sensitive than closed-form arithmetic -- the optimizer's exact convergence path, not just
# float summation order, differs slightly by platform. Two increasingly broad attempts to exclude
# this by exact key name or by widening REL_TOL module-wide each still failed on the next Linux CI
# run, on yet another key -- because the set of affected keys isn't a fixed handful, it's every
# *fitted* value these two labs record. Enumerated precisely instead, per session, by reading each
# lab's full key list end to end and classifying every key: a fit's own coefficients/p-values/
# z-stats and anything computed FROM the fit (a forecast, an RMSE against a forecast) go here; the
# *_true/_theory design constants the fit is checked against, and every value computed by ordinary
# arithmetic (ACF, realized-vol estimators, EWMA with a fixed lambda -- no iterative optimizer
# involved) do not, and stay strictly checked. Scoped by session id so this can never affect an
# unrelated session's same-named key. Each session's own live assert on the fitted quantity (e.g.
# `fit_asym.pvalues["gamma[1]"] < 0.01`, `gamma_z > N_SE`) still guards a real regression.
SESSION_NONDETERMINISTIC_KEYS = {
    "5.1": (
        "fit_asym_alpha", "fit_asym_beta", "fit_asym_gamma", "fit_asym_gamma_p", "fit_asym_nu",
        "fit_asym_omega", "fit_sym_gamma", "fit_sym_gamma_p", "persist_fit_mean", "persist_fit_se",
    ),
    "5.5": (
        "garch_alpha_hat", "garch_beta_hat", "garch_omega_hat", "garch_rmse_pp",
        "gjr_alpha_hat", "gjr_beta_hat", "gjr_gamma_hat", "gjr_omega_hat",
        "gjr_alpha_wrong_fit", "gjr_gamma_z",
        "forecast_garch_1d", "forecast_garch_60d", "forecast_gap_garch_pp",
    ),
}


def is_nondeterministic(key: str, session_id: str | None = None) -> bool:
    return (any(s in key for s in NONDETERMINISTIC_KEYS)
            or key.endswith(NONDETERMINISTIC_SUFFIXES)
            or key.startswith(NONDETERMINISTIC_PREFIXES)
            or key in SESSION_NONDETERMINISTIC_KEYS.get(session_id, ()))


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


def compare(old: dict, new: dict, session_id: str | None = None) -> list[str]:
    problems = []
    for key in sorted(set(old) | set(new)):
        if key not in new:
            problems.append(f"{key}: missing after re-run")
        elif key not in old:
            problems.append(f"{key}: new result not committed")
        elif is_nondeterministic(key, session_id):
            pass  # present on both sides; value is allowed to vary run to run (see above)
        else:
            a, b = old[key], new[key]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL):
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
        problems = compare(committed, fresh, s.id)
        for p in problems:
            print(f"  FAIL {s.id}: {p}")
        failed += bool(problems)
        if not problems:
            print(f"  ok   {s.id}: {len(fresh)} results reproduced")
    print(f"verify_labs: {len(wanted)} lab(s), {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
