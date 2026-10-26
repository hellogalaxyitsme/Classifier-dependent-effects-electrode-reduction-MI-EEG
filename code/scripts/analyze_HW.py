"""Experiments H (common-average reference) and W (1-s and 2-s windows):
full-montage accuracy and selector - full differences (pp, 90% CI)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.analysis import load, paired, subject_level  # noqa: E402

EXP = sys.argv[1]
DS = sys.argv[2].split(",")
for ds in DS:
    d = load(EXP, ds)
    if d.empty:
        continue
    s = subject_level(d)
    print(f"\n=== {EXP} {ds}: {s.subject.nunique()} subjects")
    for kr in ["rstar", "fixed"]:
        x = s[s.k_rule.isin([kr, "full"])]
        for dec in ["MDM", "FGMDM", "TSLR", "CSPLDA"]:
            w = x[x.decoder == dec].pivot_table(index="subject", columns="selector", values="acc")
            out = [f"{kr:5s} {dec:6s} full {100 * w['FULL'].mean():5.1f}"]
            for sel in ["T1", "PREC", "SH", "GEO", "RAND"]:
                if sel in w:
                    r = paired(w[sel], w["FULL"])
                    out.append(f"{sel} {r['mean']:+5.1f} [{r['lo90']:+5.1f},{r['hi90']:+5.1f}]")
            print("  ".join(out))
