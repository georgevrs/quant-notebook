"""Where things live, locally and in Colab.

The repo root is found by walking up from the current directory until `course.yaml` appears.
Outside the repo (e.g. a notebook opened in Colab) everything falls back to the working directory.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root(start: Path | None = None) -> Path | None:
    p = (start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "course.yaml").exists() and (candidate / "labs").is_dir():
            return candidate
    return None


def data_cache() -> Path:
    """Git-ignored cache for downloaded data (override with QUANTNB_CACHE)."""
    env = os.environ.get("QUANTNB_CACHE")
    if env:
        path = Path(env)
    else:
        root = repo_root()
        path = (root / "data" / "cache") if root else Path.cwd() / "quantnb_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def offline() -> bool:
    """True when network access must not be attempted (CI sets QUANTNB_OFFLINE=1)."""
    return os.environ.get("QUANTNB_OFFLINE", "").strip() not in ("", "0", "false", "False")
