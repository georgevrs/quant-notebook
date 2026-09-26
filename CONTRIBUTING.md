# Contributing to Quant Notebook

Thank you for helping. The course is only as good as its accuracy, so the most valuable
contributions are **corrections**.

## Report an erratum

Use the [erratum form](https://github.com/georgevrs/quant-notebook/issues/new?template=errata.yml):
the session, the section or figure, what is wrong, and what it should say — with a source if it is
a factual matter. Out-of-date tools, broken labs and unclear explanations count too.

## Propose a change

1. Open an issue first for anything beyond a small fix, so we can agree on the approach.
2. Read [AUTHORING.md](AUTHORING.md): it defines the voice, the page anatomy, the diagram rules,
   the lab rules and the citation standard every session follows.
3. Set up: `uv sync --all-extras` and `npm ci`.
4. Make the change, then run the checks:

   ```bash
   uv run python tools/build.py --run-labs --only <session-id>
   uv run python tools/build.py --check
   node tools/shoot.mjs site/<unit>/<session>.html     # look at the screenshots
   uv run pytest && npm test
   ```

5. Open a pull request describing what changed and why. CI runs the same checks.

## Ground rules

- Every factual claim needs a verifiable source; every number from a simulation comes from a lab.
- Never commit licensed market data (see [data/SOURCES.md](data/SOURCES.md)).
- Nothing in the course may read as personal investment advice or promise returns.
- By contributing you agree that your text is licensed under CC BY 4.0 and your code under MIT,
  as described in [LICENSE-CONTENT](LICENSE-CONTENT) and [LICENSE](LICENSE).
