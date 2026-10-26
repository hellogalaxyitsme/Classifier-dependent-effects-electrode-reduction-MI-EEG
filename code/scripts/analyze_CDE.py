"""Summaries of experiments B (count curves), C (training size), D (montage
density) and E (selection leakage).

usage: python analyze_CDE.py <exp> [datasets]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.analysis import load, paired, subject_level  # noqa: E402

pd.set_option("display.width", 250)
EXP = sys.argv[1]
DS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bnci2014_001", "cho2017", "lee2019_mi",
                                                       "schirrmeister2017"]
DECS = ["MDM", "MDM_OAS", "TSLR", "CSPLDA"]


def fmt(r):
    return f"{r['mean']:+5.1f} [{r['lo90']:+5.1f},{r['hi90']:+5.1f}]"


for ds in DS:
    d = load(EXP, ds)
    if d.empty:
        continue
    print(f"\n=== {EXP} {ds}: {d.subject.nunique()} subjects, errors {int((d.status != 'ok').sum())}")
    if EXP == "B":
        s = subject_level(d, by_k=True)
        for dec in DECS:
            w = s[s.decoder == dec].pivot_table(index="subject", columns=["selector", "k"], values="acc")
            full = w[("FULL", w["FULL"].columns[0])]
            line = {}
            for sel in ["T1", "LO", "GEO", "RAND", "PT1", "PPCA"]:
                if sel not in w.columns.get_level_values(0):
                    continue
                line[sel] = {int(k): 100 * (w[(sel, k)] - full).mean() for k in w[sel].columns}
            print(f"-- {dec}: full {100 * full.mean():.1f}%; selector - full (pp) by k")
            print(pd.DataFrame(line).T.round(1).to_string())
            # best-k-on-test inflation: per subject, max over the grid of T1 accuracy
            t1 = w["T1"]
            best = t1.max(axis=1)
            print(f"   best-k-on-test T1 - full: {100 * (best - full).mean():+.1f} pp")
    elif EXP == "C":
        s = subject_level(d)
        a = subject_level(load("A", ds)[lambda x: x.seed == 42])
        a["frac"] = 1.0
        s = pd.concat([s, a[a.selector.isin(["FULL", "T1", "PT1", "RAND"])]], ignore_index=True)
        for dec in DECS:
            print(f"-- {dec}")
            for frac in sorted(s.frac.unique()):
                x = s[(s.decoder == dec) & (s.frac == frac)]
                w = x.pivot_table(index="subject", columns=["selector", "k_rule"], values="acc")
                full = w[("FULL", "full")]
                out = [f"frac {frac:.2f}: full {100 * full.mean():5.1f}"]
                for sel, kr in [("T1", "rstar"), ("RAND", "rstar"), ("PT1", "rstar"), ("T1", "fixed"),
                                ("RAND", "fixed")]:
                    if (sel, kr) in w:
                        out.append(f"{sel}@{kr} {fmt(paired(w[(sel, kr)], full))}")
                print("   " + "  ".join(out))
    elif EXP == "D":
        s = subject_level(d)
        for mont in sorted(s.montage.unique()):
            print(f"-- montage {mont}")
            for dec in DECS:
                w = s[(s.decoder == dec) & (s.montage == mont)].pivot_table(index="subject", columns="selector",
                                                                             values="acc")
                full = w["FULL"]
                out = [f"{dec:8s} full {100 * full.mean():5.1f}"]
                for sel in ["T1", "JG", "LO", "CSPF", "BB", "RAND", "PT1"]:
                    if sel in w:
                        out.append(f"{sel} {paired(w[sel], full)['mean']:+5.1f}")
                print("   " + "  ".join(out))
    elif EXP == "E":
        s = subject_level(d)
        a = subject_level(load("A", ds)[lambda x: x.seed == 42])
        for dec in DECS:
            we = s[s.decoder == dec].pivot_table(index="subject", columns="selector", values="acc")
            wa = a[(a.decoder == dec) & a.k_rule.isin(["rstar", "full"])].pivot_table(
                index="subject", columns="selector", values="acc")
            out = [f"{dec:8s}"]
            for sel in ["T1", "CSPF", "BB"]:
                if sel in we and sel in wa:
                    out.append(f"{sel}: leak - nested {fmt(paired(we[sel], wa[sel]))}")
            print("   " + "  ".join(out))
