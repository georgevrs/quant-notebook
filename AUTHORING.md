# Authoring guide — Quant Notebook

This is the contract for writing a session. It is written for the author agents that build the
course in parallel, and it is equally the guide for human contributors. Read it all once; keep
§12 (the checklist) open while you work.

The design system comes from the **course-notebook-design** skill (installed locally at
`.claude/skills/course-notebook-design/`). Read its `references/writing-voice.md`,
`references/components.md` and `references/svg-diagrams.md` before your first page. This file
adds what is specific to Quant Notebook and overrides the skill where they differ.

The two **golden sessions** show what "done" looks like. Open them in a browser before you start:

- `site/unit01-quant-landscape/session03-research-lifecycle.html` (conceptual) with its lab
  `labs/u01/s03_research_lifecycle.py`
- `site/unit02-markets-instruments/session03-returns-pnl-leverage.html` (math, charts) with its lab
  `labs/u02/s03_returns_pnl_leverage.py`

---

## 1. Who we write for

Strong software, data and AI engineers who want to become professional quants: researchers,
traders, quant developers or independent systematic traders. They know Python, SQL, testing,
ML basics and cloud. They know **no finance**. Assume nothing about markets and explain every
convention; assume a lot about engineering and use it for analogies.

The promise is honest: the course teaches the process professionals use, not a way to get rich.
Never promise returns. Where the evidence is sobering (most traders lose, backtests overstate,
edges decay), say so plainly and cite it.

## 2. What you own, and what you must not touch

You are assigned **one session**, identified by its id (for example `4.2`). You may create and
edit only:

| Path | What |
|---|---|
| `site/unitUU-<unit-slug>/sessionSS-<slug>.html` | your page (already scaffolded) |
| `labs/uUU/sSS_<slug>.py` | your lab, percent format (already scaffolded) |
| `labs/uUU/sSS_<slug>.ipynb` | generated from the `.py` by the build — do not hand-edit |
| `labs/uUU/out/sSS.results.json`, `labs/uUU/out/sSS_*.svg` | written by running your lab |

Everything else is **read-only**: `course.yaml`, `site/assets/**`, every other page, `tools/**`,
`src/quantnb/**`, `pyproject.toml`, `uv.lock`, `package.json`. Other agents are working at the
same time; editing shared files corrupts their work.

- **Never** run `uv add`, `uv sync`, `uv lock`, `npm install` or `pip install`. The environment is
  frozen for the wave (`UV_FROZEN=1`, `UV_NO_SYNC=1`). If you need a package that is not installed,
  do without it and list it under `deps` in your handoff.
- If `quantnb` lacks a helper you need, **write it inline in your lab** (clearly, it is teaching
  code) and list it under `quantnb_requests` in your handoff. The orchestrator promotes good helpers
  into the package between waves.
- **Never** place real orders, connect a broker with real credentials, or use any MCP brokerage,
  email or messaging tool. Broker labs use paper accounts and recorded fixtures only.

## 3. The workflow

```bash
# 0. your page and lab stub already exist (the orchestrator scaffolded them)
# 1. research: canonical references + current practice (web search); decide the 6–8 sections
# 2. write the lab FIRST, run it until it passes its own asserts
uv run python labs/uUU/sSS_<slug>.py
# 3. write the page; every number you quote comes from the lab's results.json
# 4. build + check your session only (renders math, fills nav, injects charts and numbers)
uv run python tools/build.py --only U.S
# 5. look at it — screenshots at phone and desktop width, plus automatic overflow checks
node tools/shoot.mjs site/unitUU-<unit-slug>/sessionSS-<slug>.html
#    then open the PNGs in .shots/<page>/ and actually look: labels, overlaps, legibility
# 6. repeat 3–5 until check_session PASSES and shoot reports no problems
# 7. return the handoff (§13)
```

Editing math after a build: the build replaced `\( … \)` with MathML. To edit a formula, run
`node tools/render_math.mjs --unrender <page>`, edit the TeX, and rebuild. (Unrendering trims the
spaces inside the delimiters.) Do the same before search-and-replace on anything near math.

Editing from scripts: write the script to a file and run it. Do **not** put TeX or backslashes in
`python -c "…"` or `node -e "…"` strings — the shell and Windows argument parsing eat
backslashes, and `\a` or `\t` silently becomes a control character.

## 4. Page anatomy

Start from the scaffolded page (a fork of the skill's session template). Keep this rhythm:

1. **Cover** — eyebrow + h1 (generated), a 1–3 sentence **subtitle** with exactly one `<mark>` on
   the core move, the meta strip `DURATION · LEVEL · TRACKS · HANDS-ON`, the lab strip (generated),
   and the TOC.
2. **6–8 sections**, each: `h2` with chip `U.S.N` → `p.lede` → content → usually one callout.
   A reliable order: *mental model* (concept diagram, 📌 key idea) → *how it works* (the workflow
   diagram, a reference table) → *the math* (words first, then formulas) → *in code* (snippets
   identical to the lab) → *where people go wrong* (naive-vs-disciplined diagram, ⚠️ pitfall) →
   *try it* (🔧 lab callout) → close.
3. **Close** — `<hr>`, then one section: glossary, ≥ 4 "check yourself" questions, a
   `🔧 Homework (~1 h before Session X.Y)` lab callout, and `📚 Go deeper` references.

### Minimums (enforced by `tools/check_session.py`)

| Element | Minimum |
|---|---|
| Hand-drawn SVG diagrams (`figure.diagram`, not `.chart`) | 3, one of them a **workflow** (3+ boxes, 2+ arrows) |
| Tables | 2 |
| Callouts | 4, of at least 3 kinds (`key`, `tip`, `warn`, `lab`, `note`) |
| Check-yourself questions (`details.q`) | 4 |
| Glossary | exactly the terms in your session's `owns` list in `course.yaml` |
| Homework | 1 `co lab` whose title contains "Homework" |
| References in `ol.refs` | 3, each with a DOI, SSRN, arXiv, ISBN or URL id |

Data charts are welcome **in addition** to the hand-drawn diagrams, never instead of them.

### Numbering

Session `U.S` (e.g. `2.3`). Section chips `2.3.1, 2.3.2 …`. Figures `fig 2.3.1, fig 2.3.2 …`
numbered in page order across diagrams and charts. Captions: `fig 2.3.4 — what it shows, ideally
the lesson`. Refer to them in prose ("see fig 2.3.4") and to sections as `§2.3.4`.

## 5. Markup reference

Use the classes exactly as `references/components.md` documents them. Quant Notebook adds:

| You write | The build does |
|---|---|
| `\( … \)` inline math, `\[ … \]` display math | TeX → MathML with the TeX kept as an annotation |
| `<span data-lab="sharpe" data-fmt=".2f">0.37</span>` | fills the text from `results.json` (Python format spec) |
| `<figure class="diagram chart">` + `<!-- chart:growth --><!-- /chart:growth -->` + `<figcaption>` | injects `labs/uUU/out/sSS_growth.svg` |
| `<a data-xref="6.3"></a>` / `<a data-xref="6.3#s4"></a>` | "Session 6.3 — Title" link, or a "coming soon" span |
| `<a data-xref="6.3" data-style="short"></a>` | "Session 6.3" |
| `<a data-term="Sharpe ratio">Sharpe ratio</a>` | links the term to the session that owns it |
| `<div class="gloss" id="glossary"><div><b>Term</b> — definition</div>…` | adds stable `id`s to each entry |

Everything inside `<!-- nav:NAME -->…<!-- /nav:NAME -->` markers is generated. Never edit it, and
never write the literal text of a marker (or `-->`) inside an HTML comment.

Callout labels: `📌 Key idea — <the idea>`, `💡 Tip — …` / `💡 Looking ahead`,
`⚠️ Pitfall — "<the mistaken belief>"`, `🔧 Try it (N min, in the lab)`, `🔧 Homework (~1 h before
Session X.Y)`, `🧭 Margin note — …`. One `<mark>` per section at most.

## 6. Voice

Follow the skill's `writing-voice.md`: plain, warm, direct, second person, opinionated, concrete
numbers, honest about difficulty. Quant Notebook adds:

- **Engineering analogies**, stated once and reused: a backtest is a unit test that can pass by
  luck; look-ahead bias is train/test leakage; an OMS is a state machine; a kill switch is a
  circuit breaker; a holdout set is a one-time password; log returns are decibels.
- **Every number has a source.** Numbers from simulations come from your lab via `data-lab`.
  Empirical claims ("58% lower after publication") carry the citation in the text or refs.
- **Candour about evidence.** Distinguish "the evidence says" from "practitioners believe" from
  "this is contested" — and when it is contested, show both sides (see the factor-zoo debate).
- **Jurisdiction-neutral.** Use US examples when they are the clearest, label them as examples,
  and mention the EU/UK equivalent where it matters. Never give personal financial, tax or legal
  advice. Date any regulatory detail ("as of 2026").
- **Cross-reference constantly**: back to where an idea was built, forward to where it returns.

## 7. Diagrams

Follow `references/svg-diagrams.md`: palette with meaning (ultramarine = concept/doing,
pine = success/right way, sticky = checking/waiting/hands-on, coral = danger/naive way),
`viewBox="0 0 720 H"` and never width/height, labelled boxes, 11px titles and 9–11px descriptions,
italic moral under the figure. Hard-won rules:

- **Budget text width**: ≈ 6 px per character at 11 px, ≈ 5.3 px at 9.5 px. Split into lines
  rather than overflow. `shoot.mjs` flags text that escapes its box — fix every flag.
- **Marker ids unique across the page**, prefixed by section: `s1a`, `s4b`.
- **No emoji and no math markup inside SVG.** Use Unicode: σ μ Σ √ ≤ ≥ → ×. Write `&amp;` for &.
- **Draw to scale or say you didn't.** If bar lengths encode numbers, use one scale.
- Diagrams show structure the prose can't: order (workflow), parts (concept boxes), contrast
  (naive vs disciplined), proportion (weighting bar), choice (decision).
- On phones the page gives each diagram a 560 px minimum width and lets it scroll inside the
  figure, so labels stay legible. Keep important content away from the far right edge anyway.

### Data charts (`quantnb.charts`)

`line_chart`, `drawdown_chart`, `histogram`, `column_chart` — they return SVG you save with
`lab.chart(name, svg)`. Roles carry meaning: `strategy` (ultramarine, the subject), `alt1`
(orange) and `alt2` (green) for compared series, `benchmark` (grey, de-emphasised), `loss`
(coral). At most three highlighted series. Out-of-sample periods are a shaded band
(`bands=[(start, end, "out-of-sample")]`), never a line colour. Titles in plain words; the chart
states what it shows, the caption states the lesson.

## 8. Math and notation

Write math in words first, then the formula. Keep derivations short on the page and complete in
the lab or the references. Use `\lt`/`\gt` (or `&lt;`/`&gt;`) for < and >.

Shared macros (`tools/macros.tex`): `\E` (expectation), `\Prob`, `\Var`, `\Cov`, `\Corr`, `\SR`,
`\diag`, `\sign`; Temml also provides `\R`, `\argmax`, `\argmin`, `\dd`, `\tr`.

Reserved notation — use these meanings everywhere, or say explicitly that you are overloading:

| Symbol | Meaning | Symbol | Meaning |
|---|---|---|---|
| \(P_t\) | price at end of period t | \(w\), \(w_i\) | portfolio weights (vector, element) |
| \(R_t\) | simple return | \(\Sigma\) | covariance matrix (never a sum) |
| \(r_t\) | log return | \(\sum\) | summation |
| \(r_f\) | risk-free rate | \(\rho\) | correlation |
| \(\mu\) | expected (arithmetic) return | \(\beta\) | factor / market exposure |
| \(\sigma\) | volatility (standard deviation) | \(\alpha\) | return not explained by factors |
| \(\SR\) | Sharpe ratio | \(\gamma\) | risk aversion (Euler γ only where stated) |
| \(W_t\) | wealth / equity | \(\lambda\) | Kyle's lambda in Unit 14 only; else a multiplier |
| \(L\) | leverage | \(f\) | Kelly fraction (Unit 3 onward) |
| \(Y\) | years of data | \(d\) | fractional-differencing order (Unit 12 only) |
| \(T\) | number of periods | \(N\) | number of assets **or** trials — say which |

## 9. Finance conventions (set in Session 2.3 — do not deviate)

- Simple returns aggregate across assets; log returns aggregate across time.
- Returns are **total returns** (dividends reinvested) on adjusted prices unless stated.
- Annualise with the periods per year of the data: 252 trading days, 52 weeks, 12 months.
  Mean × periods, volatility × √periods.
- The Sharpe ratio uses **excess** returns and is annualised; state the risk-free rate used.
- Growth is reported as **CAGR** (geometric), never as an arithmetic average of returns.
- 1 basis point (bp) = 0.01%. Say "percentage points" for differences between percentages.
- Money: `$1,000`, thousands separators; dates ISO (`2026-09-26`) in code and tables.

## 10. Labs

- **Percent-format `.py` is the source of truth.** Keep the scaffolded header and Colab cell.
- Start with `lab = qn.Lab("U.S")` and use `lab.rng` for all randomness. Never call
  `np.random.seed` elsewhere or use unseeded generators.
- **Default to synthetic data** from `quantnb.synth` — it is legal to redistribute, runs offline,
  and you know the truth, so you can test whether a method recovers it.
- Real data only from free sources, fetched at runtime into the cache and guarded:
  `if not qn.offline(): …fetch…` with a synthetic fallback, so CI (offline) still passes.
  Never commit downloaded data. Allowed without special care: FRED public-domain series, the
  Kenneth French library (fetch, don't commit), SEC EDGAR (User-Agent from the `SEC_USER_AGENT`
  environment variable), Binance public data. Never Yahoo/yfinance in labs that must run in CI.
- **Budget: under 180 seconds on one CPU core**, no GPU, no network in CI.
- **Assert what you teach**: every key result has an `assert` with a tolerance, so a broken lab
  fails loudly instead of quietly teaching something false.
- `lab.record(key, value)` for every number the page quotes; `lab.chart(name, svg)` for every
  chart; `lab.save()` as the last cell.
- Teaching code is read more than it is run: short functions, docstrings that say what and why,
  named constants in CAPS at the top of each section.

### Library verdicts (2026)

| Use | For | Avoid / flag |
|---|---|---|
| pandas 3, polars, numpy, scipy | data, numerics | pandas-datareader (broken), `.asi8` on datetime indexes (unit varies in pandas 3) |
| DuckDB, Parquet, Arrow | research storage | ArcticDB for business use (BSL licence) |
| statsmodels, arch | econometrics, GARCH, SPA/MCS | — |
| scikit-learn, LightGBM, shap | ML | FinRL for anything but a warning |
| cvxpy, skfolio | optimisation, portfolio, CPCV | — |
| QuantLib, vollib | derivatives pricing, implied vol | py_vollib (deprecated), FinancePy (GPL), rateslib (non-commercial) |
| vectorbt | fast parameter sweeps — always scored with PBO/DSR | its Commons Clause: no selling products built on it |
| NautilusTrader | event-driven backtest = live | backtrader (unmaintained) |
| hftbacktest | order-book / market-making backtests | — |
| quantstats, alphalens-reloaded | reports, factor tear sheets | original pyfolio/empyrical/alphalens (dead) |
| alpaca-py, ib_async | paper trading | ib_insync (dead), alpaca-trade-api (deprecated) |
| mlfinlab | — | proprietary since 2020: implement AFML methods yourself |

## 11. Citations

- Cite primary sources: the paper or book that established the result, plus the best modern
  treatment. Prefer peer-reviewed papers, standard textbooks and official documentation.
- Every reference has an identifier (DOI, SSRN id, arXiv id, ISBN or a stable URL).
- **Verify every reference with a web search before citing it** — title, authors, year, venue,
  identifier. Never cite from memory alone; an invented citation is the worst error this course can
  make. If you cannot verify it, leave it out.
- Format: `Author, A. &amp; Author, B. (Year). "Title." <em>Venue</em> vol(issue), pages.
  <span class="rid">doi:…</span>`.

## 12. Before you hand back — the checklist

- [ ] `uv run python tools/build.py --only U.S` → `BUILD OK` (check_session PASS)
- [ ] `node tools/shoot.mjs <page>` → "no layout problems", and you **looked** at the PNGs
- [ ] The lab runs in under 180 s and its asserts pass
- [ ] Every number in the prose is a `data-lab` span or has a citation
- [ ] Glossary = exactly your `owns` terms; terms owned elsewhere are `data-term` links
- [ ] Every reference verified by web search; each has an id
- [ ] Cross-references back (prerequisites) and forward (where it returns)
- [ ] You edited only your own files (§2)

## 13. The handoff

End your final message with this YAML block (the orchestrator merges these between waves):

```yaml
session: "U.S"
status: done            # or: blocked
build: "BUILD OK"       # paste the last line of tools/build.py --only U.S
shoot: "no layout problems"
new_terms: []           # terms you needed that no session owns (don't define them — list them)
deps: []                # packages you wanted but were not installed
quantnb_requests: []    # helpers written inline that the package should offer (name + one line)
xrefs: []               # sessions you referenced, e.g. ["5.2", "6.7"]
sources: []             # every reference you cited, as "Author (Year) — id"
doubts: []              # anything you are unsure is correct, or had to simplify
```
