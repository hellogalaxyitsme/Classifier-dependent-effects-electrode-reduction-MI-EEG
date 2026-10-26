"""Stability of the selected channel subsets across folds and split seeds.

Nogueira, Sechidis and Brown (JMLR 2018) stability, which allows subsets of
different sizes: phi = 1 - mean_f s_f^2 / (kbar/d (1 - kbar/d)), where s_f^2
is the unbiased variance of the selection indicator of channel f over the M
subsets of one subject. Reported per selector and count rule as the mean
over subjects.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.data import PROJ  # noqa: E402

C = {"bnci2014_001": 22, "cho2017": 64, "lee2019_mi": 62, "schirrmeister2017": 128}


def nogueira(Z):
    M, d = Z.shape
    kbar = Z.sum(1).mean()
    s2 = Z.var(0, ddof=1)
    den = (kbar / d) * (1 - kbar / d)
    return 1 - s2.mean() / den if den > 0 else np.nan


rows = []
for ds, d in C.items():
    files = glob.glob(str(PROJ / "runs" / "A*" / ds / "sel_subj_*_seed_*.csv.gz"))
    if not files:
        continue
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for (sel, kr, k), g in s.groupby(["selector", "k_rule", s.k.where(s.k_rule == "fixed", -1)]):
        vals = []
        for subj, gs in g.groupby("subject"):
            Z = np.zeros((len(gs), d))
            for i, ch in enumerate(gs["channels"].astype(str)):
                Z[i, [int(v) for v in ch.split()]] = 1
            vals.append(nogueira(Z))
        rows.append(dict(dataset=ds, selector=sel, k_rule=kr, k=k, n_subj=len(vals),
                         phi=np.nanmean(vals), phi_sd=np.nanstd(vals)))
out = pd.DataFrame(rows)
print(out.pivot_table(index=["dataset", "k_rule", "k"], columns="selector", values="phi").round(2).to_string())
out.to_csv(PROJ / "results" / "stability_A.csv", index=False)
