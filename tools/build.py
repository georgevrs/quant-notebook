#!/usr/bin/env python3
"""The one build command (orchestrator-only; authors run it for their own session with --only).

    uv run python tools/build.py                       inject → math → nav → notebooks → checks
    uv run python tools/build.py --run-labs            … and first re-run every lab (.py) that exists
    uv run python tools/build.py --run-labs --only 2.3 … one session only
    uv run python tools/build.py --check               verify: nothing would change and all checks pass

Steps, in order:
  1. labs      (opt-in) run labs/uNN/sSS_*.py → labs/uNN/out/sSS.results.json + sSS_<chart>.svg
  2. inject    <!-- chart:NAME --> blocks ← labs/uNN/out/sSS_NAME.svg ;
               <span data-lab="key" data-fmt=".2f">…</span> ← results.json
  3. math      node tools/render_math.mjs (TeX → MathML, in place)
  4. nav       tools/sync_nav.py (tabs, heads, footers, hubs, cross-refs)
  5. notebooks labs/**/*.py (percent format) → .ipynb via jupytext (outputs stripped)
  6. checks    render_math --check, check_session on every existing session page
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course import ROOT, Course, Session, read_text, sub_outside_comments, write_text  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LAB_TIMEOUT = 180
CHART_RE = re.compile(r"(<!-- chart:([a-z0-9_-]+) -->)(.*?)(<!-- /chart:\2 -->)", re.S)
LAB_SPAN_RE = re.compile(r'(<span\b[^>]*\bdata-lab="([^"]+)"[^>]*>)(.*?)(</span>)', re.S)


def node() -> str:
    exe = shutil.which("node")
    if not exe:
        sys.exit("node not found on PATH")
    return exe


# ---------------------------------------------------------------------------------------------
def run_lab(s: Session) -> bool:
    if not s.lab_py.exists():
        print(f"  skip {s.id}: no lab at {s.repo_rel(s.lab_py)}")
        return True
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               QUANTNB_OFFLINE=os.environ.get("QUANTNB_OFFLINE", "1"), PYTHONUTF8="1",
               MPLBACKEND="Agg")
    print(f"  lab {s.id}: {s.repo_rel(s.lab_py)}")
    try:
        proc = subprocess.run([sys.executable, str(s.lab_py)], cwd=ROOT, env=env, timeout=LAB_TIMEOUT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        print(f"  FAIL {s.id}: lab exceeded {LAB_TIMEOUT}s")
        return False
    if proc.returncode != 0:
        print(f"  FAIL {s.id}: lab exited {proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}")
        return False
    return True


def fmt_value(value, spec: str | None) -> str:
    """Python format spec, plus a leading '$' for money: data-fmt="$,.2f" gives −$1,234.50."""
    if spec is None or isinstance(value, str):
        return str(value)
    if spec.startswith("$"):
        body = format(abs(value), spec[1:])
        neg = value < 0 and any(ch in "123456789" for ch in body)
        return ("−" if neg else "") + "$" + body
    out = format(value, spec)
    if out.startswith("-") and not any(ch in "123456789" for ch in out):
        out = out[1:]  # "-0.0%" → "0.0%"
    if out.startswith("-"):
        out = "−" + out[1:]  # typographic minus, matching hand-typed text
    return out


def inject(s: Session, text: str, problems: list[str]) -> str:
    def chart(m: re.Match) -> str:
        name = m.group(2)
        svg_path = s.lab_out / f"s{s.no:02d}_{name}.svg"
        if not svg_path.exists():
            problems.append(f"chart:{name} — {s.repo_rel(svg_path)} missing (run the lab)")
            return m.group(0)
        return f"{m.group(1)}\n{read_text(svg_path).strip()}\n{m.group(4)}"
    text = CHART_RE.sub(chart, text)

    results = None
    if s.results_json.exists():
        results = json.loads(s.results_json.read_text(encoding="utf-8"))["results"]

    def lab_num(m: re.Match) -> str:
        key = m.group(2)
        if results is None:
            problems.append(f'data-lab="{key}" but {s.repo_rel(s.results_json)} missing (run the lab)')
            return m.group(0)
        if key not in results:
            problems.append(f'data-lab="{key}" not found in results.json')
            return m.group(0)
        spec = re.search(r'\bdata-fmt="([^"]*)"', m.group(1))
        try:
            val = fmt_value(results[key], spec.group(1) if spec else None)
        except (ValueError, TypeError) as e:
            problems.append(f'data-lab="{key}": bad data-fmt ({e})')
            return m.group(0)
        return f"{m.group(1)}{val}{m.group(4)}"
    return sub_outside_comments(LAB_SPAN_RE, lab_num, text)


def notebook_for(py: Path) -> str:
    """Generate the .ipynb text for a percent-format .py (outputs empty)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / (py.stem + ".ipynb")
        subprocess.run([sys.executable, "-m", "jupytext", "--quiet", "--to", "ipynb", "--output", str(out), str(py)],
                       check=True, capture_output=True, text=True)
        nb = json.loads(out.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        cell["id"] = f"cell-{i:02d}"  # nbformat mints random ids; stable ids keep diffs clean
    nb.setdefault("metadata", {})["colab"] = {"provenance": []}
    return json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only: fail if anything would change")
    ap.add_argument("--run-labs", action="store_true", help="re-run lab scripts before injecting")
    ap.add_argument("--only", nargs="*", help="session ids to limit labs/inject/checks to (e.g. 2.3)")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    course = Course()
    sessions = [s for s in course.sessions if (not args.only or s.id in args.only)]
    failed = False
    drift: list[str] = []

    # 1. labs
    if args.run_labs and not args.check:
        print("[1/6] labs")
        for s in sessions:
            failed |= not run_lab(s)

    # 2. inject
    print("[2/6] inject charts + lab numbers")
    for s in sessions:
        if not s.path.exists():
            continue
        text = read_text(s.path)
        problems: list[str] = []
        new = inject(s, text, problems)
        for p in problems:
            print(f"  FAIL {s.rel_path}: {p}")
            failed = True
        if new != text:
            drift.append(s.rel_path)
            if not args.check:
                write_text(s.path, new)

    # With --only, every step touches only those sessions' files: parallel authors must never
    # rewrite each other's pages or notebooks.
    only_pages = [str(s.path) for s in sessions if s.path.exists()] if args.only else ["site"]
    only_flag = ["--only", *args.only] if args.only else []

    # 3. math
    print("[3/6] math")
    if not args.check and only_pages:
        r = subprocess.run([node(), "tools/render_math.mjs", *only_pages], cwd=ROOT, text=True,
                           encoding="utf-8", errors="replace", capture_output=True)
        print("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-2000:])
            failed = True

    # 4. nav
    print("[4/6] nav")
    r = subprocess.run([sys.executable, "tools/sync_nav.py"] + (["--check"] if args.check else []) + only_flag,
                       cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    print("  " + r.stdout.strip().replace("\n", "\n  "))
    if r.returncode:
        if args.check:
            drift.append("nav")
        failed = True

    # 5. notebooks
    print("[5/6] notebooks")
    lab_files = ([s.lab_py for s in sessions if s.lab_py.exists()] if args.only
                 else sorted((ROOT / "labs").rglob("*.py")))
    for py in lab_files:
        if "out" in py.parts or py.name.startswith("_"):
            continue
        want = notebook_for(py)
        nb = py.with_suffix(".ipynb")
        have = nb.read_text(encoding="utf-8") if nb.exists() else None
        if have != want:
            drift.append(nb.relative_to(ROOT).as_posix())
            if not args.check:
                write_text(nb, want)

    if args.check and drift:
        print("FAIL out of date (run tools/build.py): " + ", ".join(drift))
        failed = True

    # 6. checks
    if not args.skip_checks:
        print("[6/6] checks")
        r = subprocess.run([node(), "tools/render_math.mjs", "--check", *only_pages], cwd=ROOT, text=True,
                           encoding="utf-8", errors="replace", capture_output=True)
        if r.returncode:
            print(r.stdout[-3000:])
            failed = True
        pages = [str(s.path) for s in sessions if s.path.exists()]
        if pages:
            r = subprocess.run([sys.executable, "tools/check_session.py", *pages], cwd=ROOT, text=True,
                               encoding="utf-8", errors="replace", capture_output=True)
            tail = [ln for ln in r.stdout.splitlines() if "FAIL" in ln or "=>" in ln or "pages pass" in ln or ln.startswith("site") or ln.startswith("unit")]
            print("  " + "\n  ".join(tail))
            failed |= r.returncode != 0
    print("BUILD FAILED" if failed else "BUILD OK")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
