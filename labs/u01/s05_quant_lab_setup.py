# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1.5 · Set Up Your Quant Lab — companion lab
#
# **Quant Notebook** · Unit 1 · Session 5 · [Read the session](https://georgevrs.github.io/quant-notebook/unit01-quant-landscape/session05-quant-lab-setup.html)
#
# A reproducible workbench — uv, Python 3.12, Jupyter, DuckDB, the quantnb package, a research repo layout and a data cache.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import numpy as np
import pandas as pd

import quantnb as qn

lab = qn.Lab("1.5")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

# %% [markdown]
# ## 1 · First step
#
# Say what this step shows and why it matters.

# %%
x = rng.standard_normal(1_000)
lab.record("example_key", float(x.mean()))
assert abs(x.mean()) < 0.2, "sanity check: the sample mean of N(0,1) draws is near zero"

# %%
lab.save()
