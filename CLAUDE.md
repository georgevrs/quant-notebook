# CLAUDE.md — Quant Notebook

An open, in-depth course (GitHub Pages site + companion Python labs) that takes software, data and
AI engineers to professional quant level: 16 units, 100 sessions, 5 phases. Built in the user's
**course-notebook-design** style — offline HTML, hand-drawn inline SVG, no JavaScript.

## Where things are

- `course.yaml` — the single source of truth: every unit and session, slugs (frozen), goals,
  prerequisites, owned glossary terms, and **status** (`planned` / `doing` / `published`).
  Read it first to know where the build stands.
- `AUTHORING.md` — the contract for writing a session (read before authoring or reviewing).
- `site/` — the published site (deployed as-is). `labs/` — one lab per session.
  `src/quantnb/` — the teaching package. `tools/` — build tooling.
- The design skill is extracted (gitignored) at `.claude/skills/course-notebook-design/`.

## Commands

```bash
uv run python tools/build.py                    # inject → math → nav → notebooks → checks
uv run python tools/build.py --run-labs --only 2.3
uv run python tools/build.py --check            # CI: nothing out of date, every page passes
uv run python tools/scaffold.py session 4.2 --status doing
uv run python tools/scaffold.py status 4.2 published
node tools/shoot.mjs site/unit02-…/session03-….html   # screenshots + layout checks (system Edge)
uv run pytest && npm test                       # quantnb + tool tests
```

## How the build is organised

Waves of parallel author subagents, one session each, following `AUTHORING.md`. Between waves the
orchestrator merges handoffs, promotes helpers into `quantnb`, runs `tools/build.py`, runs a
fact-check reviewer per unit and a consistency reviewer per wave, flips statuses to `published`,
commits and pushes. Only the orchestrator edits shared files (`course.yaml`, `tools/`, `src/`,
`site/assets/`, hub pages).

## Rules

- Never trade with real money or use brokerage/email/messaging connectors (denied in
  `.claude/settings.json`). Broker labs use paper accounts and recorded fixtures.
- Never commit licensed or downloaded market data; labs default to seeded synthetic data.
- Never cite a reference without verifying it; every page number in prose comes from a lab or a
  citation.
- Write files with LF endings and no BOM; write TeX/backslash edits from script files, not
  `-c`/`-e` strings.
