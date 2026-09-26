#!/usr/bin/env python3
"""Generate every navigational fragment of the site from course.yaml.

    uv run python tools/sync_nav.py            rewrite marker regions in site/ (idempotent)
    uv run python tools/sync_nav.py --check    exit 1 if a rewrite would change anything

Only text between <!-- nav:NAME --> and <!-- /nav:NAME --> is ever replaced, plus three
deterministic normalisations of authored content:
  * cross-references   <a data-xref="6.3#s4"></a>           → link (or "coming soon" span)
  * glossary term links <a data-term="Sharpe ratio">…</a>    → link to the owning session
  * glossary entry ids  <div class="gloss"> <div><b>Term</b>  → <div id="g-term"><b>Term</b>
A page that has a marker twice, or a session page missing a required marker, is an error.
Links to sessions whose status is not "published" render as non-link <span class="soon">.
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course import (SITE, Course, Session, esc, read_text, rel_link, sub_outside_comments,  # noqa: E402
                    term_anchor, write_text)

SESSION_MARKERS = ["head", "tabs", "heading", "labstrip", "foot"]

HUB_PAGES = {
    "index.html": ("Quant Notebook · Course Home",
                   "A free, open, in-depth course that takes software, data and AI engineers to professional quant level — markets, math, research discipline, portfolios, derivatives, machine learning and production trading."),
    "start-here.html": ("Start Here · Quant Notebook",
                        "How to use Quant Notebook: who it is for, the tracks, the lab setup, how sessions work, and the disclaimer."),
    "roadmap.html": ("Roadmap · Quant Notebook",
                     "All 100 sessions of Quant Notebook — five phases, sixteen units — with goals, prerequisites and publication status."),
    "glossary.html": ("Glossary · Quant Notebook",
                      "Every term defined in Quant Notebook, with a one-line definition and a link to the session that teaches it."),
    "library.html": ("Library · Quant Notebook",
                     "The books, papers, tools and data sources behind Quant Notebook, with an honest verdict on each."),
    "404.html": ("Page not found · Quant Notebook", "This page does not exist (yet)."),
}
HUB_TABS = [("index.html", "⌂ Course home"), ("start-here.html", "🧭 Start here"),
            ("roadmap.html", "🗺 Roadmap"), ("glossary.html", "📖 Glossary")]

PHASE_COLOURS = {  # fill, stroke, title
    "ultramarine": ("#EDF0FC", "#3A56C5", "#3A56C5"),
    "pine": ("#E9F5EF", "#1F7A55", "#1F7A55"),
    "sticky": ("#FFF6C9", "#B99B22", "#8A6D00"),
    "coral": ("#FCEEE9", "#D64B2A", "#D64B2A"),
    "neutral": ("#F5F6F2", "#4A5A68", "#1F2C38"),
}


class NavError(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# marker plumbing
# ---------------------------------------------------------------------------------------------
def marker_re(name: str) -> re.Pattern:
    return re.compile(r"<!-- nav:%s -->.*?<!-- /nav:%s -->" % (re.escape(name), re.escape(name)), re.S)


def markers_in(text: str) -> list[str]:
    return re.findall(r"<!-- nav:([a-z0-9-]+) -->", text)


def fill(text: str, name: str, content: str) -> str:
    pat = marker_re(name)
    n = len(pat.findall(text))
    if n != 1:
        raise NavError(f"marker nav:{name} found {n} times (expected exactly 1)")
    block = f"<!-- nav:{name} -->\n{content.strip()}\n<!-- /nav:{name} -->"
    return pat.sub(lambda _m: block, text)


# ---------------------------------------------------------------------------------------------
# fragments
# ---------------------------------------------------------------------------------------------
def head_block(c: Course, page_rel: str, title: str, description: str, absolute_assets: bool = False) -> str:
    depth = page_rel.count("/")
    if absolute_assets:  # 404.html is served at any depth, so it needs root-absolute asset paths
        base = "/" + c.repo.split("/")[1] + "/assets/"
    else:
        base = "../" * depth + "assets/"
    url = c.canonical(page_rel)
    lines = [
        f"<title>{esc(title)}</title>",
        f'<meta name="description" content="{esc(description)}">',
    ]
    if page_rel != "404.html":
        lines.append(f'<link rel="canonical" href="{esc(url)}">')
    lines += [
        '<meta property="og:type" content="article">',
        f'<meta property="og:site_name" content="{esc(c.meta["name"])}">',
        f'<meta property="og:title" content="{esc(title)}">',
        f'<meta property="og:description" content="{esc(description)}">',
    ]
    if page_rel != "404.html":
        lines.append(f'<meta property="og:url" content="{esc(url)}">')
    lines += [
        '<meta name="twitter:card" content="summary">',
        f'<link rel="stylesheet" href="{base}brand.css">',
        f'<link rel="stylesheet" href="{base}temml.css">',
    ]
    return "\n".join(lines)


def link_or_soon(c: Course, from_rel: str, target: Session, label: str, cls: str = "") -> str:
    cls_attr = f' class="{cls}"' if cls else ""
    if target.published:
        return f'<a{cls_attr} href="{rel_link(from_rel, target.rel_path)}">{esc(label)}</a>'
    soon_cls = f"{cls} soon".strip()
    return f'<span class="{soon_cls}" title="Coming soon">{esc(label)}</span>'


def session_tabs(c: Course, s: Session) -> str:
    rel = s.rel_path
    out = ['<nav class="tabs">', f'<a href="{rel_link(rel, "index.html")}">⌂ Course home</a>']
    for sib in s.unit.sessions:
        if sib.id == s.id:
            out.append(f'<a class="active" href="{sib.filename}">{esc(sib.label)}</a>')
        else:
            out.append(link_or_soon(c, rel, sib, sib.label))
    out.append(f'<a href="{rel_link(rel, "roadmap.html")}">🗺 Roadmap</a>')
    out.append("</nav>")
    return "\n".join(out)


def session_heading(c: Course, s: Session) -> str:
    k = len(s.unit.sessions)
    return (f'<p class="eyebrow">Unit {s.unit_no} · {esc(s.unit.name)} — Session {s.no:02d} of {k:02d}</p>\n'
            f"<h1>{esc(s.title)}</h1>")


def session_labstrip(c: Course, s: Session) -> str:
    nb = s.repo_rel(s.lab_ipynb)
    py = s.repo_rel(s.lab_py)
    return ('<div class="labstrip"><b>📓 Companion lab</b>\n'
            f'<a href="{esc(c.colab(nb))}">Open in Colab</a>\n'
            f'<a href="{esc(c.github_blob(nb))}">Notebook on GitHub</a>\n'
            f'<a href="{esc(c.github_blob(py))}">Source (.py)</a>\n'
            "</div>")


def legal_line(from_rel: str) -> str:
    start = rel_link(from_rel, "start-here.html")
    return (f'<p class="legal">Educational material only — not investment, legal or tax advice. '
            f'Backtests and simulations are hypothetical. <a href="{start}#disclaimer">Disclaimer</a> · '
            f"Text CC BY 4.0 · Code MIT.</p>")


def session_foot(c: Course, s: Session) -> str:
    rel = s.rel_path
    prev, nxt = c.prev_next(s)
    home = rel_link(rel, "index.html")
    if prev is None:
        left = f'<a href="{home}">← Course home</a>'
    else:
        left = link_or_soon(c, rel, prev, f"← Prev: {prev.label}")
    if nxt is None:
        right = f'<a href="{home}">Course home →</a>'
    else:
        right = link_or_soon(c, rel, nxt, f"Next: {nxt.label} →")
    return f'<div class="foot">\n{left}\n{right}\n</div>\n{legal_line(rel)}'


def hub_tabs(current: str) -> str:
    out = ['<nav class="tabs">']
    for href, label in HUB_TABS:
        active = ' class="active"' if href == current else ""
        out.append(f'<a{active} href="{href}">{label}</a>')
    out.append("</nav>")
    return "\n".join(out)


def hub_foot(c: Course, page_rel: str) -> str:
    first = c.sessions[0]
    start = link_or_soon(c, page_rel, first, f"Start: Session {first.id} — {first.title} →")
    return f'<div class="foot">\n<a href="#top">↑ Back to top</a>\n{start}\n</div>\n{legal_line(page_rel)}'


# ---------------------------------------------------------------------------------------------
# hub page bodies
# ---------------------------------------------------------------------------------------------
def wrap_words(text: str, max_chars: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def phase_map_svg(c: Course) -> str:
    n = len(c.phases)
    x0, gap, width = 16, 12, 728
    bw = (width - gap * (n - 1)) / n
    parts = ['<figure class="diagram">',
             '<svg viewBox="0 0 760 196" xmlns="http://www.w3.org/2000/svg" style="font-family:\'IBM Plex Sans\',sans-serif">',
             '  <defs><marker id="mapA" markerWidth="7" markerHeight="7" refX="6" refY="2.5" orient="auto"><path d="M0,0 L6,2.5 L0,5 z" fill="#4A5A68"/></marker></defs>',
             '  <text x="16" y="22" font-size="10.5" fill="#1F2C38" font-weight="600">THE ARC — five phases, each ending with something you can show</text>']
    for i, ph in enumerate(c.phases):
        fill_c, stroke, title_c = PHASE_COLOURS[ph["colour"]]
        x = x0 + i * (bw + gap)
        cx = x + bw / 2
        units = [c.unit(u) for u in ph["units"]]
        n_sess = sum(len(u.sessions) for u in units)
        n_pub = sum(1 for u in units for s in u.sessions if s.published)
        parts.append(f'  <rect x="{x:.1f}" y="36" width="{bw:.1f}" height="96" rx="9" fill="{fill_c}" stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'  <text x="{cx:.1f}" y="56" font-size="10" text-anchor="middle" fill="{title_c}" font-weight="700">PHASE {ph["id"]}</text>')
        for j, line in enumerate(wrap_words(ph["name"], 20)[:2]):
            parts.append(f'  <text x="{cx:.1f}" y="{73 + j * 14}" font-size="11" text-anchor="middle" fill="{title_c}" font-weight="600">{esc(line)}</text>')
        ulabel = f"units {units[0].id}–{units[-1].id}" if len(units) > 1 else f"unit {units[0].id}"
        parts.append(f'  <text x="{cx:.1f}" y="108" font-size="9" text-anchor="middle" fill="#4A5A68">{ulabel} · {n_sess} sessions</text>')
        parts.append(f'  <text x="{cx:.1f}" y="122" font-size="8.5" text-anchor="middle" fill="#4A5A68">{n_pub} published</text>')
        for j, line in enumerate(wrap_words(ph["milestone"], 24)[:3]):
            parts.append(f'  <text x="{cx:.1f}" y="{152 + j * 12}" font-size="9" text-anchor="middle" fill="{title_c}" font-style="italic">{esc(line)}</text>')
        if i < n - 1:
            ax = x + bw + 1
            parts.append(f'  <path d="M{ax:.1f} 84 L{ax + gap - 3:.1f} 84" stroke="#4A5A68" stroke-width="1.4" marker-end="url(#mapA)"/>')
    parts.append("</svg>")
    parts.append(f'<figcaption>fig 0.1 — the five phases of {esc(c.meta["name"])}, from foundations to a professional practice</figcaption>')
    parts.append("</figure>")
    return "\n".join(parts)


def track_matrix(c: Course) -> str:
    keys = list(c.track_defs)
    head = "".join(f"<th>{esc(k)}</th>" for k in keys)
    rows = []
    for u in c.units:
        cells = "".join(f"<td>{'●' if k in u.tracks else '○'}</td>" for k in keys)
        rows.append(f"<tr><td><code>U{u.id}</code></td><td>{esc(u.name)}</td>{cells}</tr>")
    legend = " · ".join(f"<strong>{esc(k)}</strong> {esc(v['name'])}" for k, v in c.track_defs.items())
    return (f'<p>{legend}. ● = core for that track, ○ = useful context.</p>\n'
            f"<table>\n<thead><tr><th>Unit</th><th>Name</th>{head}</tr></thead>\n<tbody>\n"
            + "\n".join(rows) + "\n</tbody>\n</table>")


def home_units(c: Course) -> str:
    out = []
    for u in c.units:
        ph = c.phase_of(u)
        out.append(f'<section id="unit{u.id}">')
        out.append(f'<h2><span class="no">Unit {u.id}</span> {esc(u.name)}</h2>')
        out.append(f'<p class="lede">Phase {ph["id"]} · {esc(ph["name"])} — {len(u.sessions)} sessions · {esc(u.level)} · '
                   f'tracks {" · ".join(esc(t) for t in u.tracks)}</p>')
        out.append('<div class="cards">')
        for s in u.sessions:
            inner = (f'<span class="k">Unit {u.id} · Session {s.no}</span>\n<h3>{esc(s.title)}</h3>\n'
                     f"<p>{esc(s.goal)}</p>\n")
            if s.published:
                out.append(f'<a class="card" href="{s.rel_path}">\n{inner}<span class="tag">→ {s.filename}</span>\n</a>')
            else:
                out.append(f'<div class="card soon">\n{inner}<span class="tag">coming soon</span>\n</div>')
        out.append("</div>")
        out.append(f'<div class="co key"><span class="co-t">📌 The thread through this unit</span>\n<p>{esc(u.thread)}</p>\n</div>')
        out.append("</section>")
    return "\n".join(out)


def status_badge(pub: int, total: int, any_doing: bool) -> str:
    if pub == total:
        return '<span class="badge done">✅ published</span>'
    if pub or any_doing:
        return '<span class="badge doing">🟡 in progress</span>'
    return '<span class="badge todo">⬜ planned</span>'


def home_status(c: Course) -> str:
    rows = []
    for u in c.units:
        pub = sum(s.published for s in u.sessions)
        doing = any(s.status == "doing" for s in u.sessions)
        ph = c.phase_of(u)
        rows.append(f"<tr><td><code>U{u.id}</code></td><td>{esc(u.name)}</td><td>{pub} of {len(u.sessions)}</td>"
                    f"<td>Phase {ph['id']}</td><td>{status_badge(pub, len(u.sessions), doing)}</td></tr>")
    total_pub = sum(s.published for s in c.sessions)
    return ("<table>\n<thead><tr><th>Unit</th><th>Name</th><th>Sessions</th><th>Phase</th><th>Status</th></tr></thead>\n<tbody>\n"
            + "\n".join(rows) + "\n</tbody>\n</table>\n"
            f"<p><strong>{total_pub} of {len(c.sessions)}</strong> sessions published.</p>")


def home_counts(c: Course) -> str:
    return (f'<span>DURATION <b>{esc(c.meta["duration"])}</b></span>\n'
            f"<span>UNITS <b>{len(c.units)}</b></span>\n"
            f"<span>SESSIONS <b>{len(c.sessions)}</b></span>\n"
            f'<span>LEVEL <b>{esc(c.meta["level"])}</b></span>')


def roadmap_body(c: Course) -> str:
    out = []
    for ph in c.phases:
        out.append(f'<section id="phase{ph["id"]}">')
        out.append(f'<h2><span class="no">Phase {ph["id"]}</span> {esc(ph["name"])}</h2>')
        out.append(f'<p class="lede">Milestone: <strong>{esc(ph["milestone"])}</strong>.</p>')
        for uid in ph["units"]:
            u = c.unit(uid)
            out.append(f'<h3 id="u{u.id}">Unit {u.id} · {esc(u.name)}</h3>')
            out.append(f"<p>{esc(u.thread)}</p>")
            out.append("<table>\n<thead><tr><th>Session</th><th>What you walk away with</th><th>Builds on</th><th>Status</th></tr></thead>\n<tbody>")
            for s in u.sessions:
                title = link_or_soon(c, "roadmap.html", s, f"{s.id} · {s.title}")
                pre = ", ".join(link_or_soon(c, "roadmap.html", c.by_id[p], p) for p in s.prereqs) or "—"
                badge = {"published": '<span class="badge done">✅ published</span>',
                         "doing": '<span class="badge doing">🟡 in progress</span>'}.get(
                    s.status, '<span class="badge todo">⬜ planned</span>')
                out.append(f"<tr><td>{title}</td><td>{esc(s.goal)}</td><td>{pre}</td><td>{badge}</td></tr>")
            out.append("</tbody>\n</table>")
        out.append("</section>")
    return "\n".join(out)


GLOSS_ENTRY_RE = re.compile(r'<div(?: id="[^"]*")?><b>(.*?)</b>\s*[—-]\s*(.*?)</div>', re.S)


def collect_definitions(c: Course) -> dict[str, tuple[str, Session]]:
    """term (lower) → (definition html, owning session), from published pages' glossaries."""
    defs: dict[str, tuple[str, Session]] = {}
    for s in c.sessions:
        if not (s.published and s.path.exists()):
            continue
        text = read_text(s.path)
        m = re.search(r'<div class="gloss"[^>]*>(.*?)\n</div>', text, re.S)
        if not m:
            continue
        for term, definition in GLOSS_ENTRY_RE.findall(m.group(1)):
            defs[html.unescape(re.sub(r"<[^>]+>", "", term)).strip().lower()] = (definition.strip(), s)
    return defs


def glossary_body(c: Course) -> str:
    defs = collect_definitions(c)
    entries = sorted(c.term_owner.items(), key=lambda kv: kv[0].lstrip("'\"").lower())
    by_letter: dict[str, list[str]] = {}
    display = {t.lower(): t for s in c.sessions for t in s.owns}
    for key, owner in entries:
        term = display[key]
        letter = term[0].upper() if term[0].isalpha() else "#"
        anchor = term_anchor(term)
        if key in defs and owner.published:
            definition, _ = defs[key]
            link = f'<a href="{owner.rel_path}#{anchor}">Session {owner.id}</a>'
            by_letter.setdefault(letter, []).append(f'<div id="{anchor}"><b>{esc(term)}</b> — {definition} <small>({link})</small></div>')
        else:
            by_letter.setdefault(letter, []).append(
                f'<div id="{anchor}"><b>{esc(term)}</b> — <span class="xref soon">defined in Session {owner.id} · {esc(owner.title)} (coming soon)</span></div>')
    out = [f"<p><strong>{sum(1 for k in defs)}</strong> of {len(entries)} terms defined so far. Every term is defined once, "
           "by the session that teaches it.</p>"]
    for letter in sorted(by_letter):
        out.append(f'<h3 id="letter-{letter.lower() if letter != "#" else "sym"}">{letter}</h3>')
        out.append('<div class="gloss">\n' + "\n".join(by_letter[letter]) + "\n</div>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# authored-content normalisation: cross-refs, term links, glossary ids
# ---------------------------------------------------------------------------------------------
XREF_RE = re.compile(r'<(a|span)\b([^>]*?)\bdata-xref="([^"]+)"([^>]*)>(.*?)</\1>', re.S)
TERM_RE = re.compile(r'<(a|span)\b([^>]*?)\bdata-term="([^"]+)"([^>]*)>(.*?)</\1>', re.S)


def _attr(attrs: str, name: str) -> str | None:
    m = re.search(r'\b%s="([^"]*)"' % re.escape(name), attrs)
    return m.group(1) if m else None


def resolve_xrefs(c: Course, page_rel: str, text: str, problems: list[str]) -> str:
    def repl(m: re.Match) -> str:
        attrs = m.group(2) + m.group(4)
        ref = m.group(3)
        inner = m.group(5)
        sid, _, frag = ref.partition("#")
        target = c.by_id.get(sid)
        if target is None:
            problems.append(f"unknown cross-reference data-xref=\"{ref}\"")
            return m.group(0)
        style = _attr(attrs, "data-style") or "full"
        auto = _attr(attrs, "data-auto") is not None or not inner.strip()
        if auto:
            label = {"short": f"Session {target.id}", "title": target.title}.get(style, f"Session {target.id} — {target.title}")
            label = esc(label)
        else:
            label = inner
        extra = (f' data-style="{style}"' if style != "full" else "") + (' data-auto=""' if auto else "")
        if target.published or target.rel_path == page_rel:
            href = rel_link(page_rel, target.rel_path) if target.rel_path != page_rel else ""
            href += f"#{frag}" if frag else ""
            return f'<a class="xref" href="{href or "#top"}" data-xref="{ref}"{extra}>{label}</a>'
        return f'<span class="xref soon" data-xref="{ref}"{extra} title="Coming soon">{label}</span>'
    return sub_outside_comments(XREF_RE, repl, text)


def resolve_terms(c: Course, page_rel: str, text: str, problems: list[str]) -> str:
    def repl(m: re.Match) -> str:
        term = html.unescape(m.group(3))  # attribute may carry entities, e.g. data-term="P&amp;L"
        inner = m.group(5) or esc(term)
        owner = c.term_owner.get(term.lower())
        if owner is None:
            problems.append(f'unknown glossary term data-term="{term}" (not owned by any session in course.yaml)')
            return m.group(0)
        anchor = term_anchor(term)
        if owner.rel_path == page_rel:
            return f'<a class="term" href="#{anchor}" data-term="{esc(term)}">{inner}</a>'
        if owner.published:
            return f'<a class="term" href="{rel_link(page_rel, owner.rel_path)}#{anchor}" data-term="{esc(term)}">{inner}</a>'
        return f'<span class="term soon" data-term="{esc(term)}" title="Defined in Session {owner.id} (coming soon)">{inner}</span>'
    return sub_outside_comments(TERM_RE, repl, text)


def normalise_gloss_ids(text: str) -> str:
    def fix_block(m: re.Match) -> str:
        body = re.sub(r'<div(?: id="[^"]*")?><b>(.*?)</b>',
                      lambda e: f'<div id="{term_anchor(html.unescape(re.sub(r"<[^>]+>", "", e.group(1))))}"><b>{e.group(1)}</b>',
                      m.group(2))
        return m.group(1) + body + m.group(3)
    return sub_outside_comments(re.compile(r'(<div class="gloss" id="glossary">)(.*?)(\n</div>)', re.S), fix_block, text)


# ---------------------------------------------------------------------------------------------
# page processors
# ---------------------------------------------------------------------------------------------
def sync_session(c: Course, s: Session, text: str, problems: list[str]) -> str:
    present = markers_in(text)
    for name in SESSION_MARKERS:
        if name not in present:
            raise NavError(f"missing marker nav:{name}")
    rel = s.rel_path
    text = fill(text, "head", head_block(c, rel, f"{s.id} · {s.title}", s.goal))
    text = fill(text, "tabs", session_tabs(c, s))
    text = fill(text, "heading", session_heading(c, s))
    text = fill(text, "labstrip", session_labstrip(c, s))
    text = fill(text, "foot", session_foot(c, s))
    text = resolve_xrefs(c, rel, text, problems)
    text = resolve_terms(c, rel, text, problems)
    text = normalise_gloss_ids(text)
    return text


def sync_hub(c: Course, rel: str, text: str, problems: list[str]) -> str:
    title, desc = HUB_PAGES[rel]
    present = set(markers_in(text))
    if "head" in present:
        text = fill(text, "head", head_block(c, rel, title, desc, absolute_assets=(rel == "404.html")))
    if "hubtabs" in present:
        text = fill(text, "hubtabs", hub_tabs(rel))
    if "home-counts" in present:
        text = fill(text, "home-counts", home_counts(c))
    if "home-map" in present:
        text = fill(text, "home-map", phase_map_svg(c))
    if "home-tracks" in present:
        text = fill(text, "home-tracks", track_matrix(c))
    if "home-units" in present:
        text = fill(text, "home-units", home_units(c))
    if "home-status" in present:
        text = fill(text, "home-status", home_status(c))
    if "roadmap" in present:
        text = fill(text, "roadmap", roadmap_body(c))
    if "glossary" in present:
        text = fill(text, "glossary", glossary_body(c))
    if "hubfoot" in present:
        text = fill(text, "hubfoot", hub_foot(c, rel))
    text = resolve_xrefs(c, rel, text, problems)
    text = resolve_terms(c, rel, text, problems)
    return text


def git_lastmod(path: Path) -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)], capture_output=True,
                             text=True, cwd=SITE.parent, check=False).stdout.strip()
        return out or date.today().isoformat()
    except OSError:
        return date.today().isoformat()


def sitemap(c: Course) -> str:
    urls = []
    for rel in ["index.html", "start-here.html", "roadmap.html", "glossary.html", "library.html"]:
        if (SITE / rel).exists():
            urls.append((c.canonical(rel), git_lastmod(SITE / rel)))
    for s in c.sessions:
        if s.published and s.path.exists():
            urls.append((c.canonical(s.rel_path), git_lastmod(s.path)))
    body = "\n".join(f"  <url><loc>{esc(u)}</loc><lastmod>{d}</lastmod></url>" for u, d in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + body + "\n</urlset>\n")


def run(check: bool, only: list[str] | None = None) -> int:
    """Sync every page, or — with `only` — just those session pages (no hubs, no sitemap).

    `only` exists for parallel authors: each rewrites nothing but their own page.
    """
    c = Course()
    changed, failed = [], False
    targets: list[tuple[Path, str]] = []
    for s in c.sessions:
        if s.path.exists() and (not only or s.id in only):
            targets.append((s.path, s.rel_path))
    if not only:
        for rel in HUB_PAGES:
            if (SITE / rel).exists():
                targets.append((SITE / rel, rel))
        # any stray html page under site/ that is neither a session nor a hub is a mistake
        known = {t[0].resolve() for t in targets}
        for p in SITE.rglob("*.html"):
            if p.resolve() not in known:
                print(f"FAIL {p.relative_to(SITE).as_posix()}: not a course.yaml session or known hub page")
                failed = True
    for path, rel in targets:
        text = read_text(path)
        problems: list[str] = []
        try:
            s = c.by_rel_path.get(rel)
            new = sync_session(c, s, text, problems) if s else sync_hub(c, rel, text, problems)
        except NavError as e:
            print(f"FAIL {rel}: {e}")
            failed = True
            continue
        for p in problems:
            print(f"FAIL {rel}: {p}")
            failed = True
        if new != text:
            changed.append(rel)
            if not check:
                write_text(path, new)
    # sitemap.xml is a build artefact (gitignored): its lastmod dates come from git history,
    # so committing it would make it stale after every commit. CI regenerates it before deploy.
    if not check and not only:
        write_text(SITE / "sitemap.xml", sitemap(c))
    verb = "would change" if check else "updated"
    print(f"sync_nav: {len(changed)} file(s) {verb}" + (": " + ", ".join(changed) if changed else ""))
    if check and changed:
        return 1
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if anything would change")
    ap.add_argument("--only", nargs="*", help="session ids: sync only these pages (no hubs, no sitemap)")
    a = ap.parse_args()
    sys.exit(run(a.check, a.only))


if __name__ == "__main__":
    main()
