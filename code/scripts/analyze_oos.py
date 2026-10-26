"""Preregistered analysis of the out-of-sample confirmation (PREREG_OOS.md).

usage: python analyze_oos.py
Writes results/oos_tests.csv and results/oos_pooled.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gaps import load_safe  # noqa: E402
from mech.analysis import subject_level  # noqa: E402
from mech.data import PROJ  # noqa: E402

SETS = ["brandl2020", "chang2025", "guttmannflury2025_mi", "weibo2014", "grossewentrup2009"]
# id: (a, b, null value, direction); a and b are (decoder, selector, k_rule)
HYP = {
    "O1": (("MDM", "PREC", "innercv"), ("MDM", "FULL", "full"), 0., "greater"),
    "O2": (("MDM", "M22", "montage"), ("MDM", "FULL", "full"), 0., "greater"),
    "O3": (("TSLR", "M22", "montage"), ("TSLR", "FULL", "full"), -2., "greater"),
    "O4": (("TSLR", "PREC", "innercv"), ("TSLR", "FULL", "full"), -2., "greater"),
    "O5": (("MDM", "PREC", "innercv"), ("MDM", "PREC", "rstar"), 0., "greater"),
    "O6": (("MDM", "PREC", "innercv"), ("TSLR", "FULL", "full"), 0., "less"),
}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.
    for i, j in enumerate(order):
        run = max(run, min(1., (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def wide(ds):
    d = load_safe("OOS", ds)
    if d.empty:
        return None
    s = subject_level(d).groupby(["subject", "decoder", "selector", "k_rule"])["acc"].mean().reset_index()
    return s.pivot_table(index="subject", columns=["decoder", "selector", "k_rule"], values="acc") * 100


def main():
    rows = []
    tabs = {ds: wide(ds) for ds in SETS}
    for hid, (a, b, null, side) in HYP.items():
        for ds, w in tabs.items():
            if w is None or a not in w or b not in w:
                rows.append(dict(hyp=hid, dataset=ds, evaluable=False))
                continue
            x = (w[a] - w[b]).dropna()
            n, m, se = len(x), x.mean(), x.std(ddof=1) / np.sqrt(len(x))
            t = (m - null) / se
            p = 1 - stats.t.cdf(t, n - 1) if side == "greater" else stats.t.cdf(t, n - 1)
            h = stats.t.ppf(0.95, n - 1) * se
            row = dict(hyp=hid, dataset=ds, evaluable=True, n=n, mean=m, se=se, lo90=m - h, hi90=m + h, p=p)
            if hid in ("O3", "O4"):
                row["p_margin1"] = 1 - stats.t.cdf((m + 1.) / se, n - 1)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for hid, g in out[out.evaluable].groupby("hyp"):
        out.loc[g.index, "p_holm"] = holm(g.p.values)
    out["supported"] = out.evaluable & (out.p_holm < 0.05)
    pooled = []
    for hid, g in out[out.evaluable].groupby("hyp"):
        y, v = g["mean"].values, g["se"].values ** 2
        wf = 1 / v
        mf = np.sum(wf * y) / wf.sum()
        Q = np.sum(wf * (y - mf) ** 2)
        k = len(y)
        tau2 = max(0., (Q - (k - 1)) / (wf.sum() - np.sum(wf ** 2) / wf.sum())) if k > 1 else 0.
        wr = 1 / (v + tau2)
        mr, se = np.sum(wr * y) / wr.sum(), np.sqrt(1 / wr.sum())
        null, side = HYP[hid][2], HYP[hid][3]
        right_side = mr > null if side == "greater" else mr < null
        majority = g.supported.sum() > k / 2
        pooled.append(dict(hyp=hid, datasets=k, supported=int(g.supported.sum()), pooled=mr,
                           lo95=mr - 1.96 * se, hi95=mr + 1.96 * se, tau=np.sqrt(tau2),
                           I2=max(0., (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.,
                           verdict="confirmed" if (majority and right_side) else "not confirmed"))
    pooled = pd.DataFrame(pooled)
    res = PROJ / "results"
    out.to_csv(res / "oos_tests.csv", index=False)
    pooled.to_csv(res / "oos_pooled.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    print()
    print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
