"""Aggregation and subject-level statistics for the mechanism experiments."""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from scipy import stats

from .data import PROJ

RUNS = PROJ / "runs"
KEYS = ["exp", "dataset", "montage", "frac", "selector", "k_rule", "decoder"]


def load(exp, dataset):
    files = sorted(glob.glob(str(RUNS / exp / dataset / "subj_*_seed_*.csv.gz")))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["montage"] = d["montage"].fillna("native")
    return d


def subject_level(d, by_k=False):
    """Accuracy per subject: pooled over folds within (seed, sub_draw, draw),
    then averaged over random draws, subsample draws and split seeds."""
    keys = KEYS + (["k"] if by_k else [])
    g = d.groupby(keys + ["subject", "seed", "sub_draw", "draw"], dropna=False)[["n_correct", "n_test"]].sum()
    g["acc"] = g["n_correct"] / g["n_test"]
    g = g.reset_index()
    g = g.groupby(keys + ["subject", "seed", "sub_draw"], dropna=False)["acc"].mean().reset_index()
    g = g.groupby(keys + ["subject", "seed"], dropna=False)["acc"].mean().reset_index()
    n_seeds = g.groupby(keys + ["subject"], dropna=False)["seed"].nunique().rename("n_seeds")
    g = g.groupby(keys + ["subject"], dropna=False)["acc"].mean().reset_index()
    return g.merge(n_seeds.reset_index(), on=keys + ["subject"])


def paired(a, b):
    """Paired subject-level comparison of accuracy series a - b (in pp)."""
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    dlt = 100 * (m["a"] - m["b"])
    n = len(dlt)
    if n < 3:
        return dict(n=n, mean=np.nan)
    se = dlt.std(ddof=1) / np.sqrt(n)
    t90 = stats.t.ppf(0.95, n - 1)
    t, p = stats.ttest_1samp(dlt, 0.)
    try:
        pw = stats.wilcoxon(dlt).pvalue
    except ValueError:
        pw = np.nan
    return dict(n=n, mean=dlt.mean(), lo90=dlt.mean() - t90 * se, hi90=dlt.mean() + t90 * se,
                p=p, p_wilcoxon=pw, dz=dlt.mean() / dlt.std(ddof=1) if dlt.std(ddof=1) > 0 else np.nan,
                frac_pos=float(np.mean(dlt > 0)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.
    for i, idx in enumerate(o):
        run = max(run, (len(p) - i) * p[idx])
        adj[idx] = min(run, 1.)
    return adj


def wide(s, index_cols=("dataset", "montage", "frac", "decoder", "subject"), col="selector",
         k_rule=None):
    x = s if k_rule is None else s[s["k_rule"].isin([k_rule, "full"])]
    return x.pivot_table(index=list(index_cols), columns=col, values="acc")
