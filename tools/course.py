"""Load course.yaml and answer every "where does X live / what is X called" question.

Shared by scaffold.py, sync_nav.py, check_session.py and build.py so that paths, numbering and
URLs are computed in exactly one place.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
LABS = ROOT / "labs"
COURSE_YAML = ROOT / "course.yaml"

LEVEL_DURATION = "1.5 h"


def esc(text: str) -> str:
    """HTML-escape text for element content and attribute values."""
    return html.escape(str(text), quote=True)


def slug_to_ident(slug: str) -> str:
    return slug.replace("-", "_")


def term_anchor(term: str) -> str:
    """Stable anchor id for a glossary term: 'Sharpe ratio' -> 'g-sharpe-ratio'."""
    s = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
    return f"g-{s}"


@dataclass
class Session:
    id: str
    slug: str
    title: str
    short: str
    goal: str
    prereqs: list[str]
    owns: list[str]
    deps: list[str]
    status: str
    unit: "Unit" = field(repr=False)
    index: int = 0  # position in course order (0-based)

    @property
    def unit_no(self) -> int:
        return int(self.id.split(".")[0])

    @property
    def no(self) -> int:
        return int(self.id.split(".")[1])

    @property
    def published(self) -> bool:
        return self.status == "published"

    # ---- site paths -------------------------------------------------------------------------
    @property
    def rel_path(self) -> str:
        """Path relative to site/, e.g. 'unit02-markets-instruments/session03-returns-pnl-leverage.html'."""
        return f"{self.unit.dirname}/session{self.no:02d}-{self.slug}.html"

    @property
    def path(self) -> Path:
        return SITE / self.rel_path

    @property
    def filename(self) -> str:
        return f"session{self.no:02d}-{self.slug}.html"

    # ---- lab paths --------------------------------------------------------------------------
    @property
    def lab_stem(self) -> str:
        return f"s{self.no:02d}_{slug_to_ident(self.slug)}"

    @property
    def lab_dir(self) -> Path:
        return LABS / f"u{self.unit_no:02d}"

    @property
    def lab_py(self) -> Path:
        return self.lab_dir / f"{self.lab_stem}.py"

    @property
    def lab_ipynb(self) -> Path:
        return self.lab_dir / f"{self.lab_stem}.ipynb"

    @property
    def lab_out(self) -> Path:
        return self.lab_dir / "out"

    @property
    def results_json(self) -> Path:
        return self.lab_out / f"s{self.no:02d}.results.json"

    def repo_rel(self, p: Path) -> str:
        return p.relative_to(ROOT).as_posix()

    @property
    def tracks(self) -> list[str]:
        return self.unit.tracks

    @property
    def level(self) -> str:
        return self.unit.level

    @property
    def label(self) -> str:
        """'2.3 · Returns & Leverage' — the tab / footer label."""
        return f"{self.id} · {self.short}"


@dataclass
class Unit:
    id: int
    slug: str
    name: str
    level: str
    tracks: list[str]
    thread: str
    sessions: list[Session] = field(default_factory=list)

    @property
    def dirname(self) -> str:
        return f"unit{self.id:02d}-{self.slug}"


class Course:
    def __init__(self, path: Path = COURSE_YAML):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.raw = raw
        self.meta = raw["course"]
        self.track_defs = raw["tracks"]
        self.phases = raw["phases"]
        self.units: list[Unit] = []
        order = 0
        for u in raw["units"]:
            unit = Unit(id=u["id"], slug=u["slug"], name=u["name"], level=u["level"],
                        tracks=list(u["tracks"]), thread=u["thread"])
            for s in u["sessions"]:
                unit.sessions.append(Session(
                    id=str(s["id"]), slug=s["slug"], title=s["title"], short=s["short"],
                    goal=s["goal"], prereqs=[str(p) for p in s["prereqs"]], owns=list(s["owns"]),
                    deps=list(s["deps"]), status=s["status"], unit=unit, index=order))
                order += 1
            self.units.append(unit)

    # ---- lookups ----------------------------------------------------------------------------
    @cached_property
    def sessions(self) -> list[Session]:
        return [s for u in self.units for s in u.sessions]

    @cached_property
    def by_id(self) -> dict[str, Session]:
        return {s.id: s for s in self.sessions}

    @cached_property
    def by_rel_path(self) -> dict[str, Session]:
        return {s.rel_path: s for s in self.sessions}

    @cached_property
    def term_owner(self) -> dict[str, Session]:
        return {t.lower(): s for s in self.sessions for t in s.owns}

    def unit(self, uid: int) -> Unit:
        return next(u for u in self.units if u.id == uid)

    def phase_of(self, unit: Unit) -> dict:
        return next(p for p in self.phases if unit.id in p["units"])

    def prev_next(self, s: Session) -> tuple[Session | None, Session | None]:
        i = s.index
        prev = self.sessions[i - 1] if i > 0 else None
        nxt = self.sessions[i + 1] if i + 1 < len(self.sessions) else None
        return prev, nxt

    # ---- urls -------------------------------------------------------------------------------
    @property
    def site_url(self) -> str:
        return self.meta["site_url"].rstrip("/") + "/"

    @property
    def repo(self) -> str:
        return self.meta["repo"]

    @property
    def branch(self) -> str:
        return self.meta.get("branch", "main")

    def github_blob(self, repo_rel: str) -> str:
        return f"https://github.com/{self.repo}/blob/{self.branch}/{repo_rel}"

    def colab(self, repo_rel: str) -> str:
        return f"https://colab.research.google.com/github/{self.repo}/blob/{self.branch}/{repo_rel}"

    def canonical(self, site_rel: str) -> str:
        return self.site_url + ("" if site_rel == "index.html" else site_rel)


def rel_link(from_site_rel: str, to_site_rel: str) -> str:
    """Relative href from one page (path relative to site/) to another."""
    from_parts = from_site_rel.split("/")[:-1]
    to_parts = to_site_rel.split("/")
    common = 0
    while common < len(from_parts) and common < len(to_parts) - 1 and from_parts[common] == to_parts[common]:
        common += 1
    ups = [".."] * (len(from_parts) - common)
    return "/".join(ups + to_parts[common:]) or to_parts[-1]


_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def sub_outside_comments(pattern: re.Pattern, repl, text: str) -> str:
    """re.sub, but never inside HTML comments (authoring notes quote the very markup we rewrite)."""
    out, last = [], 0
    for m in _COMMENT_RE.finditer(text):
        out.append(pattern.sub(repl, text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(pattern.sub(repl, text[last:]))
    return "".join(out)


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 with LF line endings and no BOM (Windows-safe, diff-stable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text.replace("\r\n", "\n"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")
