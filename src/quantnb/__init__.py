"""Quant Notebook companion package.

Small, readable implementations used by the course labs. Every module is taught in a session;
the docstring of each says which one. Prefer reading the code to calling it blindly.
"""
from . import charts, returns, synth
from .lab import Lab
from .paths import data_cache, offline, repo_root

__version__ = "0.1.0"
__all__ = ["Lab", "charts", "returns", "synth", "data_cache", "offline", "repo_root", "__version__"]
