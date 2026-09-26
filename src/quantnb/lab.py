"""The `Lab` object every companion lab starts with.

    lab = Lab("2.3")                 # seeds numpy's global RNG and gives you lab.rng
    lab.record("sharpe", 0.73)       # numbers the session page quotes (via data-lab spans)
    lab.chart("equity", svg)         # SVG produced by quantnb.charts, injected into the page
    lab.save()                       # writes labs/uNN/out/sSS.results.json (+ the charts)

Inside the repo, output goes to labs/uNN/out/. Anywhere else (Colab), to ./out/.
The page quotes numbers from results.json, so prose and code can never drift apart.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .paths import repo_root

DEFAULT_SEED = 20260926


class Lab:
    def __init__(self, session: str, seed: int = DEFAULT_SEED):
        self.session = session
        unit, no = (int(x) for x in session.split("."))
        self.unit, self.no = unit, no
        self.seed = seed
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.results: dict[str, float | int | str | bool] = {}
        self.charts: dict[str, str] = {}
        root = repo_root()
        self.out_dir = (root / "labs" / f"u{unit:02d}" / "out") if root else Path.cwd() / "out"

    def record(self, key: str, value) -> None:
        """Store a number (or short string) the session page will quote."""
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} is not finite: {value}")
        self.results[key] = value

    def chart(self, name: str, svg: str) -> None:
        if not svg.lstrip().startswith("<svg"):
            raise ValueError("chart() expects an <svg> string from quantnb.charts")
        self.charts[name] = svg

    def save(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"s{self.no:02d}.results.json"
        payload = {"session": self.session, "seed": self.seed, "results": self.results}
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        for name, svg in self.charts.items():
            with open(self.out_dir / f"s{self.no:02d}_{name}.svg", "w", encoding="utf-8", newline="\n") as fh:
                fh.write(svg.strip() + "\n")
        print(f"saved {len(self.results)} results and {len(self.charts)} charts → {self.out_dir}")
        return path
