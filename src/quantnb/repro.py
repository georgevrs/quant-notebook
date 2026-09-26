"""Reproducibility helpers (taught in Session 1.5).

Fingerprints let you prove two runs — or two machines — produced the same data, independent of
file format or compression. Integer data fingerprints are exact everywhere; float data is rounded
to a fixed number of significant digits first, because summation order can differ across BLAS
builds in the last bits.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def fingerprint(arr, length: int = 16) -> str:
    """Short SHA-256 of an integer array's values (exact and platform-independent)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:length]


def content_fingerprint(df: pd.DataFrame, sig_digits: int = 12, length: int = 16) -> str:
    """SHA-256 of a DataFrame's content in canonical order (sorted columns and index).

    Independent of how it was stored (Parquet codec, CSV, row order). Floats are rounded to
    `sig_digits` significant digits so harmless last-bit differences don't change the hash.
    """
    d = df.sort_index(axis=1).sort_index(axis=0)
    h = hashlib.sha256()
    h.update("|".join(map(str, d.columns)).encode())
    h.update("|".join(map(str, d.index)).encode())
    for col in d.columns:
        v = d[col].to_numpy()
        if np.issubdtype(v.dtype, np.floating):
            v = np.array([float(f"{x:.{sig_digits}g}") if np.isfinite(x) else x for x in v])
            h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
        else:
            h.update("|".join(map(str, v)).encode())
    return h.hexdigest()[:length]
