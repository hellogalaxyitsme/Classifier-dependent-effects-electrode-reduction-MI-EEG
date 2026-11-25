"""Shared-subset interactions, fixed-count random controls and per-participant
TS-LR changes with M22 (Supplementary Sections S6 and S9, Tables S9 and S14,
Fig. S1a).

usage: python analyze_controls.py [outdir]

Reads runs/{KSEL,OOS,rescaled/CONF,B} and writes
  shared_subset_interactions.csv  MDM-minus-TS-LR interaction when both decoders
                                  receive the same PREC subset (50 % of the
                                  montage, or 16 electrodes)
  fixed_count_random.csv          PREC/LO and GEO against random subsets of the
                                  same fixed size (confirmatory k = 16;
                                  development k-grid)
  m22_tslr_participants.csv       per-participant TS-LR change with M22
  m22_tslr_distribution.csv       proportions and quartiles of that change

Participant accuracy pools the test trials of the five folds within a split
seed, session and random draw, then averages over draws, seeds and sessions.
Intervals are unadjusted two-sided 95 % t intervals across participants.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ = Path(os.environ.get("P", Path(__file__).resolve().parents[2]))
RUNS = PROJ / "runs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ / "results"
OUT.mkdir(parents=True, exist_ok=True)

KSEL = ["cho2017", "lee2019_mi", "schirrmeister2017", "bnci2014_001",
        "dreyer2023", "physionetmi", "yang2025", "stieger2021"]
OOS = ["brandl2020", "guttmannflury2025_mi", "weibo2014", "grossewentrup2009"]
CONF = ["dreyer2023", "physionetmi", "yang2025", "stieger2021"]


def load(exp, ds):
    files = sorted(glob.glob(str(RUNS / exp / ds / "subj_*_seed_*.csv.gz")))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:      # participant without an evaluable session
            continue
    d = pd.concat(frames, ignore_index=True)
    d = d[d["status"] == "ok"]
    d["session"] = d["montage"].astype(str)
    return d


def participant_acc(d, keys):
    """Accuracy (%) per participant for every configuration in `keys`."""
    levels = ["subject", "session", "seed", "draw"]
    g = d.groupby(keys + levels, dropna=False)[["n_correct", "n_test"]].sum()
    g["acc"] = 100 * g["n_correct"] / g["n_test"]
    g = g.reset_index()
    for drop in ["draw", "seed", "session"]:      # average over draws, then seeds, then sessions
        levels.remove(drop)
        g = g.groupby(keys + levels, dropna=False)["acc"].mean().reset_index()
    return g


def ci95(x):
    x = pd.Series(x, dtype=float).dropna()
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, mean=m, lo95=m - h, hi95=m + h)


def wide(d, sel_rules):
    """Participant accuracy, one column per (selector, k_rule, decoder)."""
    keep = d[d[["selector", "k_rule"]].apply(tuple, axis=1).isin(sel_rules)]
    a = participant_acc(keep, ["selector", "k_rule", "decoder"])
    a["col"] = a["selector"] + "|" + a["k_rule"] + "|" + a["decoder"]
    return a.pivot(index="subject", columns="col", values="acc")


def shared_subset():
    rows = []
    for exp, dss in [("KSEL", KSEL), ("OOS", OOS)]:
        for ds in dss:
            d = load(exp, ds)
            if d.empty:
                continue
            w = wide(d, [("FULL", "full"), ("PREC", "frac50"), ("PREC", "fixed16")])
            for rule in ["frac50", "fixed16"]:
                need = [f"PREC|{rule}|MDM", f"PREC|{rule}|TSLR", "FULL|full|MDM", "FULL|full|TSLR"]
                if not all(c in w for c in need):
                    continue
                dm = w[f"PREC|{rule}|MDM"] - w["FULL|full|MDM"]
                dt = w[f"PREC|{rule}|TSLR"] - w["FULL|full|TSLR"]
                r = dict(source=exp, dataset=ds, k_rule=rule, mdm_change=dm.mean(), tslr_change=dt.mean())
                r.update({f"interaction_{k}": v for k, v in ci95(dm - dt).items()})
                rows.append(r)
    # the same comparison at k = 16 within the confirmatory protocol (all sessions)
    for ds in CONF:
        d = load("rescaled/CONF", ds)
        d = d[(d["k_rule"] != "fixed") | (d["k"] == 16)]
        w = wide(d, [("FULL", "full"), ("PREC", "fixed")])
        dm = w["PREC|fixed|MDM"] - w["FULL|full|MDM"]
        dt = w["PREC|fixed|TSLR"] - w["FULL|full|TSLR"]
        r = dict(source="CONF_normalised", dataset=ds, k_rule="fixed16", mdm_change=dm.mean(), tslr_change=dt.mean())
        r.update({f"interaction_{k}": v for k, v in ci95(dm - dt).items()})
        rows.append(r)
    return pd.DataFrame(rows)


def fixed_count_random():
    rows = []
    for ds in CONF:
        d = load("rescaled/CONF", ds)
        d = d[(d["k_rule"] == "full") | ((d["k_rule"] == "fixed") & (d["k"] == 16))]
        w = wide(d, [("FULL", "full"), ("RAND", "fixed"), ("PREC", "fixed"), ("GEO", "fixed")])
        for dec in ["MDM", "TSLR"]:
            rnd = w[f"RAND|fixed|{dec}"]
            for name, a, b in [("RAND-FULL", rnd, w[f"FULL|full|{dec}"]),
                               ("PREC-RAND", w[f"PREC|fixed|{dec}"], rnd),
                               ("GEO-RAND", w[f"GEO|fixed|{dec}"], rnd)]:
                rows.append(dict(source="CONF_normalised", dataset=ds, k=16, decoder=dec, contrast=name, **ci95(a - b)))
    grid = {"cho2017": [24], "lee2019_mi": [24], "schirrmeister2017": [64, 24]}
    for ds, ks in grid.items():
        d = load("B", ds)
        for k in ks:
            dk = d[(d["k_rule"] == "full") | ((d["k_rule"] == "grid") & (d["k"] == k))]
            w = wide(dk, [("FULL", "full"), ("RAND", "grid"), ("LO", "grid"), ("GEO", "grid")])
            for dec in ["MDM", "TSLR"]:
                rnd = w[f"RAND|grid|{dec}"]
                for name, a, b in [("RAND-FULL", rnd, w[f"FULL|full|{dec}"]),
                                   ("LO-RAND", w[f"LO|grid|{dec}"], rnd),
                                   ("GEO-RAND", w[f"GEO|grid|{dec}"], rnd)]:
                    rows.append(dict(source="B", dataset=ds, k=k, decoder=dec, contrast=name, **ci95(a - b)))
    return pd.DataFrame(rows)


def m22_tslr():
    parts = []
    for exp, dss in [("rescaled/CONF", CONF), ("OOS", ["brandl2020", "guttmannflury2025_mi", "weibo2014"])]:
        for ds in dss:
            d = load(exp, ds)
            w = wide(d, [("FULL", "full"), ("M22", "montage")])
            ch = (w["M22|montage|TSLR"] - w["FULL|full|TSLR"]).dropna()
            parts.append(pd.DataFrame(dict(dataset=ds, subject=ch.index, tslr_change=ch.values)))
    p = pd.concat(parts, ignore_index=True)
    summ = []
    for ds, x in p.groupby("dataset", sort=False)["tslr_change"]:
        n = len(x)
        se = x.std(ddof=1) / np.sqrt(n)
        summ.append(dict(dataset=ds, n=n, gain_gt3=100 * np.mean(x > 3), within3=100 * np.mean(x.abs() <= 3),
                         loss_gt3=100 * np.mean(x < -3), loss_gt5=100 * np.mean(x < -5),
                         median=x.median(), q25=x.quantile(0.25), q75=x.quantile(0.75),
                         mean=x.mean(), lo90=x.mean() - stats.t.ppf(0.95, n - 1) * se))
    return p, pd.DataFrame(summ)


if __name__ == "__main__":
    shared_subset().to_csv(OUT / "shared_subset_interactions.csv", index=False)
    fixed_count_random().to_csv(OUT / "fixed_count_random.csv", index=False)
    p, s = m22_tslr()
    p.to_csv(OUT / "m22_tslr_participants.csv", index=False)
    s.to_csv(OUT / "m22_tslr_distribution.csv", index=False)
    print("written to", OUT)
