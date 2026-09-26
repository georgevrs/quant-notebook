# Quant Notebook

**From engineer to quant — markets, math, models and machines.**

A free, open, in-depth course that takes software, data and AI engineers to professional quant
level: researcher, trader, quant developer or independent systematic trader. One hundred
illustrated sessions, each with a companion Python lab you can run in Colab with one click.

**→ Read the course: [georgevrs.github.io/quant-notebook](https://georgevrs.github.io/quant-notebook/)**

[![site](https://github.com/georgevrs/quant-notebook/actions/workflows/site.yml/badge.svg)](https://github.com/georgevrs/quant-notebook/actions/workflows/site.yml)
[![labs](https://github.com/georgevrs/quant-notebook/actions/workflows/labs.yml/badge.svg)](https://github.com/georgevrs/quant-notebook/actions/workflows/labs.yml)
[![content: CC BY 4.0](https://img.shields.io/badge/content-CC%20BY%204.0-lightgrey.svg)](LICENSE-CONTENT)
[![code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

---

## Why this course

You already build systems that ingest data, train models and ship to production. Quant finance
asks you to do the same in the one domain where the data is adversarial, the signal is faint and
every mistake has a price. What you usually lack is not engineering — it is market structure,
financial mathematics, and above all the **research discipline** that stops you from fooling
yourself with a backtest.

So this course is ordered differently from most: markets, data and statistics first; then a whole
unit on backtesting, validation and overfitting statistics **before** factors, strategies and
machine learning; then portfolios and risk, derivatives, rates, microstructure, production trading
systems and careers. It is also honest: no course can make you profitable, and it says so — what it
teaches is the process professionals use to find an edge, prove it is real, size it, execute it and
survive being wrong.

## The curriculum

| Phase | Units |
| --- | --- |
| **1 · Foundations** | 1 The Quant Landscape · 2 Markets & Instruments · 3 The Quant Math Toolkit · 4 Financial Data Engineering |
| **2 · Research Craft** | 5 Statistics & Time Series of Returns · 6 Backtesting & Research Discipline · 7 Asset Pricing & Factor Investing · 8 Strategy Families |
| **3 · Portfolio, Risk & Derivatives** | 9 Portfolio Construction & Risk · 10 Derivatives & Volatility · 11 Rates, Credit & Macro |
| **4 · Machines & Markets** | 12 Machine Learning for Trading · 13 NLP, LLMs & Alternative Data · 14 Microstructure, Execution & Market Making · 15 Production Trading Systems |
| **5 · The Professional** | 16 Career, Interviews & Your Own Book |

The full list of sessions, with goals and prerequisites, is on the
[roadmap](https://georgevrs.github.io/quant-notebook/roadmap.html). The course is published in
waves; the [course home](https://georgevrs.github.io/quant-notebook/) shows what is live.

Every session has: a mental model built with hand-drawn diagrams, the math (rendered as native
MathML — no JavaScript, works offline), code identical to its lab, a naive-vs-disciplined
comparison, scenario questions with answers, homework, and verified references.

## Run the labs locally

```bash
git clone https://github.com/georgevrs/quant-notebook
cd quant-notebook
uv sync                      # core; add --all-extras for derivatives, ML, backtesting, live trading
uv run python labs/u02/s03_returns_pnl_leverage.py
```

Or open any lab in Colab from the **📓 Companion lab** strip on its session page. Labs default to
seeded synthetic data, so they run offline, anywhere, and reproduce the numbers on the page.

## Repository layout

```text
site/        the published course (plain HTML + one stylesheet; deployed to GitHub Pages as-is)
labs/        one companion lab per session (percent-format .py → generated .ipynb)
src/quantnb/ the small teaching package the labs use
tools/       build tooling: math rendering, navigation, validation, screenshots, scaffolding
course.yaml  the course structure — single source of truth for every session
AUTHORING.md how sessions are written (voice, anatomy, diagrams, labs, citations)
```

`uv run python tools/build.py --check` verifies that every page meets the standard, every number
matches its lab and nothing is out of date.

## Contributing

Errata are the most valuable contribution: use the
[erratum form](https://github.com/georgevrs/quant-notebook/issues/new?template=errata.yml).
For anything larger, read [CONTRIBUTING.md](CONTRIBUTING.md) and [AUTHORING.md](AUTHORING.md).

## Licence, disclaimer, citation

- Text and diagrams: [CC BY 4.0](LICENSE-CONTENT). Code — labs, `quantnb`, tools, stylesheet and
  code blocks in pages: [MIT](LICENSE). Fonts and third-party components: see [NOTICE](NOTICE).
- **Educational material only — not investment, legal or tax advice.** Backtests are hypothetical.
  Read the full [disclaimer](DISCLAIMER.md).
- To cite the course, use GitHub's "Cite this repository" button ([CITATION.cff](CITATION.cff)).
