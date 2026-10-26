"""Summaries of experiment A (matched-count selector panel).

usage: python analyze_A.py [datasets]
Prints, per dataset and decoder, the full-montage accuracy and the
subject-level paired difference (selector - full, pp) for each selector at
k = r* and at fixed k, followed by the matched-count contrasts for MDM.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.analysis import load, paired, subject_level  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
DS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["bnci2014_001", "cho2017", "lee2019_mi",
                                                       "schirrmeister2017"]
SELS = ["T1", "JG", "LO", "SH", "PREC", "VIF", "NS", "VAR", "CSPF", "CSPP", "BB", "GEO", "RAND", "PT1",
        "PJG", "PPCA"]

for ds in DS:
    d = load("A", ds)
    if d.empty:
        continue
    for extra in ("A2", "A3", "A4"):
        d2 = load(extra, ds)
        if not d2.empty:
            keep_full = extra == "A4"          # FgMDM is evaluated only in A4
            d2 = d2[((d2.selector != "FULL") | keep_full) & d2.subject.isin(d.subject.unique())].assign(exp="A")
            d = pd.concat([d, d2], ignore_index=True)
    print(f"\n=== {ds}: {d.subject.nunique()} subjects, seeds {sorted(d.seed.unique())}, "
          f"errors {int((d.status != 'ok').sum())}")
    s = subject_level(d)
    for k_rule in ["rstar", "fixed"]:
        ks = sorted(d.loc[d.k_rule == k_rule, "k"].unique()) if k_rule == "fixed" else [None]
        for k in ks:
            if k_rule == "fixed":
                sk = subject_level(d[(d.k_rule == "full") | ((d.k_rule == "fixed") & (d.k == k))], by_k=False)
            else:
                sk = s[s.k_rule.isin(["rstar", "full"])]
            rows = []
            for dec in ["MDM", "MDM_OAS", "FGMDM", "TSLR", "CSPLDA"]:
                w = sk[sk.decoder == dec].pivot_table(index="subject", columns="selector", values="acc")
                if "FULL" not in w:
                    continue
                row = dict(decoder=dec, full=100 * w["FULL"].mean())
                for sel in SELS:
                    if sel in w:
                        r = paired(w[sel], w["FULL"])
                        row[sel] = f"{r['mean']:+.1f}"
                rows.append(row)
            label = "k=r*" if k_rule == "rstar" else f"k={k}"
            print(f"-- {label}: selector - full (pp)")
            print(pd.DataFrame(rows).set_index("decoder").to_string(float_format=lambda v: f"{v:.1f}"))
    # matched-count contrasts at r* for MDM and TSLR
    for dec in ["MDM", "MDM_OAS", "FGMDM", "TSLR"]:
        w = s[(s.decoder == dec) & s.k_rule.isin(["rstar", "full"])].pivot_table(
            index="subject", columns="selector", values="acc")
        print(f"-- {dec} at k=r*: T1 vs comparators (pp, 90% CI, p)")
        for comp in ["SH", "LO", "PREC", "VIF", "NS", "VAR", "GEO", "RAND", "JG", "CSPF", "CSPP", "BB",
                     "PJG", "PPCA"]:
            if comp in w:
                r = paired(w["T1"], w[comp])
                print(f"   T1 - {comp:5s} {r['mean']:+5.1f} [{r['lo90']:+5.1f}, {r['hi90']:+5.1f}] "
                      f"p={r['p']:.2g} n={r['n']}")
