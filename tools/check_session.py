#!/usr/bin/env python3
"""Check Quant Notebook session pages against the visual and content standard.

    uv run python tools/check_session.py site                 every session page in course.yaml
    uv run python tools/check_session.py site/unit02-…/session03-….html

Forked from the notebook design system's check_session.py. The skill's minimums are unchanged;
Quant Notebook adds: hand-drawn diagrams counted separately from data charts, no raw TeX, every
<math> annotated, a companion-lab strip whose links map to real repo files, data-lab numbers
present in results.json, glossary terms = the session's `owns` list, 3+ references with ids,
disclaimer + head meta, and exact-case relative links. Exit 1 on any FAIL. Standard library +
PyYAML only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course import ROOT, SITE, Course, Session, read_text  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MINIMUMS = {  # element: (minimum, human label)
    "diagram": (3, "hand-drawn SVG diagrams (figure.diagram, not .chart)"),
    "table": (2, "tables"),
    "callout": (4, "callouts (.co)"),
    "quiz": (4, "check-yourself questions (details.q)"),
    "gloss": (1, "glossary (.gloss)"),
    "handson": (1, "hands-on close (a Homework .co lab, or a lab section of 2+ .co lab steps)"),
    "refs": (3, "references in 📚 Go deeper (ol.refs li)"),
}
SKIP_TEX = {"pre", "code", "svg", "math", "script", "style", "title", "textarea"}
REF_ID_RE = re.compile(r"(doi:|10\.\d{4,}/|SSRN|arXiv|ISBN|https?://)", re.I)


class Scan(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.counts = {k: 0 for k in MINIMUMS}
        self.charts = 0
        self.homework = 0
        self.labs = 0
        self.workflows = 0
        self.callout_types: set[str] = set()
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self.ids: dict[str, int] = {}
        self.stack: list[str] = []
        self.in_svg = 0
        self.svg_is_chart = False
        self.svg_rects = 0
        self.svg_arrows = 0
        self.fig_is_chart = False
        self.lab_text = ""
        self.div_depth = 0
        self.lab_depth: int | None = None
        self.figcaption: str | None = None
        self.stylesheet = False
        self.in_math = 0
        self.math_annotated = False
        self.math_count = 0
        self.skip_depth = 0
        self.hrefs: list[str] = []
        self.labstrip_hrefs: list[str] = []
        self.in_labstrip = 0
        self.data_lab: list[str] = []
        self.gloss_terms: list[str] = []
        self.in_gloss = 0
        self.gloss_b: str | None = None
        self.in_refs = False
        self.ref_text = ""
        self.ref_items: list[str] = []
        self.in_ref_li = False
        self.meta: dict[str, str] = {}
        self.canonical = False
        self.legal = False
        self.sections: list[dict] = []
        self.expect_lede = False
        self.h1 = ""
        self.in_h1 = False

    # -- tags ---------------------------------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if "id" in a:
            self.ids[a["id"]] = self.ids.get(a["id"], 0) + 1
        if tag in SKIP_TEX:
            self.skip_depth += 1
        if tag == "link":
            if "brand.css" in (a.get("href") or ""):
                self.stylesheet = True
            if a.get("rel") == "canonical":
                self.canonical = True
        if tag == "meta":
            key = a.get("name") or a.get("property")
            if key:
                self.meta[key] = a.get("content") or ""
        if tag == "img":
            self.problems.append("uses <img> — draw it as inline SVG instead")
        if tag == "script" and a.get("src"):
            self.problems.append(f"loads external script {a['src']} — pages must work offline")
        if tag == "script":
            self.problems.append("contains <script> — pages are JavaScript-free")
        if tag == "figure" and "diagram" in cls:
            self.fig_is_chart = "chart" in cls
            if self.fig_is_chart:
                self.charts += 1
            else:
                self.counts["diagram"] += 1
        if tag == "table":
            self.counts["table"] += 1
        if tag == "details" and "q" in cls:
            self.counts["quiz"] += 1
        if tag == "figcaption":
            self.figcaption = ""
        if tag == "h1":
            self.in_h1 = True
        if tag == "section":
            self.sections.append({"id": a.get("id", "?"), "marks": 0, "lede": None})
            self.expect_lede = True
        if tag == "p" and self.expect_lede and self.sections:
            self.sections[-1]["lede"] = "lede" in cls
            self.expect_lede = False
        if tag == "mark" and self.sections:
            self.sections[-1]["marks"] += 1
        if tag == "a":
            href = a.get("href")
            if href:
                self.hrefs.append(href)
                if self.in_labstrip:
                    self.labstrip_hrefs.append(href)
        if tag == "span" and "data-lab" in a:
            self.data_lab.append(a["data-lab"])
        if tag == "p" and "legal" in cls:
            self.legal = True
        if tag == "ol" and "refs" in cls:
            self.in_refs = True
        if tag == "li" and self.in_refs:
            self.in_ref_li = True
            self.ref_text = ""
        if tag == "div":
            self.div_depth += 1
            if "labstrip" in cls:
                self.in_labstrip = self.div_depth
            if "gloss" in cls:
                self.counts["gloss"] += 1
                self.in_gloss = self.div_depth
            if "co" in cls:
                self.counts["callout"] += 1
                self.callout_types.update(c for c in cls if c != "co")
                if "lab" in cls:
                    self.labs += 1
                    self.lab_depth = self.div_depth
                    self.lab_text = ""
        if tag == "b" and self.in_gloss:
            self.gloss_b = ""
        if tag == "math":
            self.in_math += 1
            if self.in_math == 1:
                self.math_count += 1
                self.math_annotated = False
        if tag == "annotation" and self.in_math and a.get("encoding") == "application/x-tex":
            self.math_annotated = True
        if tag == "svg":
            if self.in_svg == 0:
                self.svg_rects = self.svg_arrows = 0
                self.svg_is_chart = self.fig_is_chart
                if "viewbox" not in {k.lower() for k in a}:
                    self.problems.append("an <svg> has no viewBox")
                if "width" in a or "height" in a:
                    self.problems.append("an <svg> sets width/height — use viewBox only so it scales on phones")
            self.in_svg += 1
        if self.in_svg:
            if tag in ("rect", "polygon"):
                self.svg_rects += 1
            if "marker-end" in a or tag in ("path", "line"):
                self.svg_arrows += 1

    def handle_endtag(self, tag):
        if tag in SKIP_TEX and self.skip_depth:
            self.skip_depth -= 1
        if tag == "svg":
            self.in_svg -= 1
            if self.in_svg == 0 and not self.svg_is_chart and self.svg_rects >= 3 and self.svg_arrows >= 2:
                self.workflows += 1
        if tag == "figure":
            self.fig_is_chart = False
        if tag == "figcaption" and self.figcaption is not None:
            if not re.match(r"\s*fig\s+[\w.]+\s+—", self.figcaption):
                self.problems.append(f"caption not in 'fig S.N — …' form: {self.figcaption.strip()[:60]!r}")
            self.figcaption = None
        if tag == "h1":
            self.in_h1 = False
        if tag == "math":
            if self.in_math == 1 and not self.math_annotated:
                self.problems.append("a <math> element has no TeX annotation (render it with tools/render_math.mjs)")
            self.in_math -= 1
        if tag == "b" and self.gloss_b is not None:
            self.gloss_terms.append(self.gloss_b.strip())
            self.gloss_b = None
        if tag == "li" and self.in_ref_li:
            self.ref_items.append(self.ref_text)
            self.in_ref_li = False
        if tag == "ol" and self.in_refs:
            self.in_refs = False
            self.counts["refs"] += len(self.ref_items)
        if tag == "div":
            if self.lab_depth == self.div_depth:
                if "homework" in self.lab_text.lower():
                    self.homework += 1
                self.lab_depth = None
            if self.in_labstrip == self.div_depth:
                self.in_labstrip = 0
            if self.in_gloss == self.div_depth:
                self.in_gloss = 0
            self.div_depth -= 1

    def handle_data(self, data):
        if self.figcaption is not None:
            self.figcaption += data
        if self.lab_depth is not None:
            self.lab_text += data
        if self.gloss_b is not None:
            self.gloss_b += data
        if self.in_ref_li:
            self.ref_text += data
        if self.in_h1:
            self.h1 += data
        if not self.skip_depth and not self.in_math and re.search(r"\\[(\[]", data):
            snippet = data.strip().replace("\n", " ")[:50]
            self.problems.append(f"raw TeX outside code (run tools/render_math.mjs): {snippet!r}")

    def handle_comment(self, data):
        pass


# -------------------------------------------------------------------------------------------------
def exact_case_exists(path: Path) -> bool:
    """os.path.exists is case-insensitive on Windows/macOS; GitHub Pages is not."""
    try:
        rel = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return path.exists()
    cur = ROOT
    for part in rel.parts:
        if not cur.is_dir() or part not in os.listdir(cur):
            return False
        cur = cur / part
    return True


def github_blob_to_path(url: str, course: Course) -> Path | None:
    prefix = f"https://github.com/{course.repo}/blob/{course.branch}/"
    colab = f"https://colab.research.google.com/github/{course.repo}/blob/{course.branch}/"
    for p in (prefix, colab):
        if url.startswith(p):
            return ROOT / unquote(url[len(p):])
    return None


def check(path: Path, course: Course) -> bool:
    text = read_text(path)
    s = Scan()
    s.feed(text)
    s.counts["handson"] = 1 if (s.homework or s.labs >= 2) else 0
    rel = path.resolve().relative_to(SITE.resolve()).as_posix()
    session: Session | None = course.by_rel_path.get(rel)
    ok = True
    print(f"\n{rel}")

    def fail(msg):
        nonlocal ok
        ok = False
        print(f"  FAIL {msg}")

    for key, (minimum, label) in MINIMUMS.items():
        n = s.counts[key]
        status = "ok  " if n >= minimum else "FAIL"
        ok &= n >= minimum
        print(f"  {status} {n:>2} / {minimum}+  {label}")
    print(f"  info {s.charts:>2} data charts · {s.math_count} formulas · {len(s.data_lab)} lab numbers")
    if not s.workflows:
        fail("no workflow diagram among the hand-drawn diagrams (3+ boxes joined by 2+ arrows)")
    mix = s.callout_types & {"key", "tip", "warn", "lab", "note"}
    if len(mix) < 3:
        fail(f"callouts not mixed — only {sorted(mix)} (need 3+ kinds)")
    leftovers = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    if leftovers:
        fail(f"unfilled tokens: {', '.join(leftovers)}")
    dupes = sorted(i for i, n in s.ids.items() if n > 1)
    if dupes:
        fail(f"duplicate ids (arrow markers vanish): {', '.join(dupes)}")
    if not s.stylesheet:
        fail("no link to brand.css")
    for prob in dict.fromkeys(s.problems):
        fail(prob)

    # ---- quant notebook rules -------------------------------------------------------------
    for key in ("description", "og:title", "og:description"):
        if not s.meta.get(key):
            fail(f"missing <meta {key}> (run tools/sync_nav.py)")
    if not s.canonical:
        fail("missing canonical link (run tools/sync_nav.py)")
    if not s.legal:
        fail("missing disclaimer line p.legal (run tools/sync_nav.py)")
    for i, sec in enumerate(s.sections):
        if sec["marks"] > 1:
            print(f"  warn §{sec['id']} has {sec['marks']} <mark>s — one per section at most")
        if sec["lede"] is False:
            print(f"  warn §{sec['id']} does not open with a p.lede")
    for item in s.ref_items:
        if not REF_ID_RE.search(item):
            fail(f"reference without DOI/SSRN/arXiv/ISBN/URL id: {item.strip()[:70]!r}")

    # links: relative, exact case, existing
    for href in dict.fromkeys(s.hrefs):
        u = urlparse(href)
        if u.scheme in ("http", "https", "mailto") or href.startswith("#"):
            continue
        if href.startswith("/"):
            fail(f"root-absolute link breaks under the Pages subpath: {href}")
            continue
        target = (path.parent / unquote(u.path)).resolve() if u.path else path
        if not exact_case_exists(target):
            fail(f"broken or wrong-case link: {href}")

    if session is not None:
        if s.h1.strip() != session.title:
            fail(f"h1 {s.h1.strip()!r} ≠ course.yaml title {session.title!r} (run tools/sync_nav.py)")
        # lab strip → real files
        if len(s.labstrip_hrefs) < 2:
            fail("missing companion-lab strip (run tools/sync_nav.py)")
        for href in s.labstrip_hrefs:
            p = github_blob_to_path(href, course)
            if p is None or not exact_case_exists(p):
                fail(f"lab link does not map to a repo file: {href}")
        # data-lab numbers → results.json
        if s.data_lab:
            if not session.results_json.exists():
                fail(f"page quotes lab numbers but {session.repo_rel(session.results_json)} does not exist — run the lab")
            else:
                results = json.loads(session.results_json.read_text(encoding="utf-8"))["results"]
                for key in s.data_lab:
                    if key not in results:
                        fail(f'data-lab="{key}" not in {session.results_json.name}')
        # glossary ↔ owns
        owned = {t.lower() for t in session.owns}
        defined = {t.lower() for t in s.gloss_terms}
        for t in s.gloss_terms:
            if t.lower() not in owned:
                owner = course.term_owner.get(t.lower())
                if owner:
                    fail(f"glossary defines '{t}', owned by Session {owner.id} — link it with data-term instead")
                else:
                    fail(f"glossary defines '{t}', which is not in this session's `owns` list")
        missing = [t for t in session.owns if t.lower() not in defined]
        if missing:
            fail(f"owned terms missing from the glossary: {', '.join(missing)}")

    print("  => PASS" if ok else "  => needs work")
    return ok


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    course = Course()
    paths: list[Path] = []
    for arg in argv:
        p = Path(arg)
        if p.is_dir():
            paths += [sess.path for sess in course.sessions if sess.path.exists()]
        else:
            paths.append(p)
    results = [check(p, course) for p in dict.fromkeys(paths)]
    print(f"\n{sum(results)}/{len(results)} pages pass")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
