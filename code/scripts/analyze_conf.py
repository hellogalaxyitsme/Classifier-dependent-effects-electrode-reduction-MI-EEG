"""Confirmatory tests H1-H6 of PREREGISTRATION.md.

usage: python analyze_conf.py
Writes results/confirmatory_tests.csv and prints the decision table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.analysis import holm, load, subject_level  # noqa: E402
from mech.data import PROJ  # noqa: E402

import os  # noqa: E402

DENSE = os.environ.get("CONF_DENSE", "physionetmi,stieger2021,yang2025").split(",")
SPARSE = os.environ.get("CONF_SPARSE", "dreyer2023").split(",")
OUT_FILE = os.environ.get("CONF_OUT", "confirmatory_tests.csv")


def subj_table(ds):
    d = load("CONF", ds)
    if d.empty:
        return None
    s = subject_level(d)                                        # per session
    s = s[s.k_rule.isin(["rstar", "full", "montage"])]
    s = s.groupby(["subject", "decoder", "selector"])["acc"].mean().reset_index()   # average sessions
    return s.pivot_table(index="subject", columns=["decoder", "selector"], values="acc") * 100


def diff(w, dec, a, b):
    return (w[(dec, a)] - w[(dec, b)]).dropna()


def ci90(x):
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.95, n - 1) * se
    return x.mean(), x.mean() - h, x.mean() + h


def one_sided_p(x, mu=0.):
    t, p = stats.ttest_1samp(x, mu)
    return p / 2 if t > 0 else 1 - p / 2


rows = []
tabs = {ds: subj_table(ds) for ds in DENSE + SPARSE}
for ds, w in tabs.items():
    if w is None:
        continue
    n = len(w)
    for dec in ["MDM", "MDM_OAS", "FGMDM", "TSLR", "CSPLDA"]:
        for sel in ["T1", "SH", "PREC", "BB", "CSPF", "GEO", "RAND", "M22"]:
            if (dec, sel) in w:
                m, lo, hi = ci90(diff(w, dec, sel, "FULL"))
                rows.append(dict(dataset=ds, n=n, test="gain_vs_full", decoder=dec, selector=sel,
                                 mean=m, lo90=lo, hi90=hi, p=one_sided_p(diff(w, dec, sel, "FULL"))))
    for comp in ["SH", "PREC"]:                                             # H1
        m, lo, hi = ci90(diff(w, "MDM", "T1", comp))
        rows.append(dict(dataset=ds, n=n, test="H1", decoder="MDM", selector=f"T1-{comp}", mean=m,
                         lo90=lo, hi90=hi, verdict="supported" if (lo > -1 and hi < 1) else "not supported"))
    for comp in ["BB", "CSPF"]:                                             # H4
        m, lo, hi = ci90(diff(w, "MDM", "PREC", comp))
        rows.append(dict(dataset=ds, n=n, test="H4", decoder="MDM", selector=f"PREC-{comp}", mean=m,
                         lo90=lo, hi90=hi, verdict="supported" if lo > -1 else "not supported"))
    for dec in ["TSLR", "FGMDM"]:                                           # H3 (second part)
        m, lo, hi = ci90(diff(w, dec, "PREC", "FULL"))
        rows.append(dict(dataset=ds, n=n, test="H3b", decoder=dec, selector="PREC-FULL", mean=m,
                         lo90=lo, hi90=hi, verdict="supported" if hi < 1 else "not supported"))
    dd = (diff(w, "MDM", "PREC", "FULL") - diff(w, "TSLR", "PREC", "FULL")).dropna()   # H3 (first part)
    m, lo, hi = ci90(dd)
    rows.append(dict(dataset=ds, n=n, test="H3a", decoder="MDM-TSLR", selector="gain difference", mean=m,
                     lo90=lo, hi90=hi, p=one_sided_p(dd)))
out = pd.DataFrame(rows)
# Holm families over the dense datasets: H2 (MDM PREC gain), H3a, H5 (MDM M22 gain)
for test, mask in (("H2", (out.test == "gain_vs_full") & (out.decoder == "MDM") & (out.selector == "PREC")),
                   ("H3a", out.test == "H3a"),
                   ("H5", (out.test == "gain_vs_full") & (out.decoder == "MDM") & (out.selector == "M22"))):
    sub = out[mask & out.dataset.isin(DENSE)]
    if len(sub):
        adj = holm(sub["p"].values)
        out.loc[sub.index, "family"] = test
        out.loc[sub.index, "p_holm"] = adj
        out.loc[sub.index, "verdict"] = np.where(adj < 0.05, "supported", "not supported")
# H6: sparse-control gain smaller than every dense gain and within +-2 pp
g = out[(out.test == "gain_vs_full") & (out.decoder == "MDM") & (out.selector == "PREC")].set_index("dataset")
for sp in SPARSE:
    if sp not in g.index:
        continue
    dense_min = g.loc[[d for d in DENSE if d in g.index], "mean"].min() if any(d in g.index for d in DENSE) else np.nan
    r = g.loc[sp]
    ok = (r["mean"] < dense_min) and (r["lo90"] > -2) and (r["hi90"] < 2)
    out = pd.concat([out, pd.DataFrame([dict(dataset=sp, test="H6", decoder="MDM", selector="PREC-FULL",
                                             mean=r["mean"], lo90=r["lo90"], hi90=r["hi90"],
                                             verdict="supported" if ok else "not supported")])])
(PROJ / "results").mkdir(exist_ok=True)
out.to_csv(PROJ / "results" / OUT_FILE, index=False)
pd.set_option("display.width", 250)
print(out[out.test != "gain_vs_full"].round(3).to_string(index=False))
print(out[(out.test == "gain_vs_full") & out.decoder.isin(["MDM", "FGMDM", "TSLR"])].pivot_table(
    index=["dataset", "decoder"], columns="selector", values="mean").round(1).to_string())
