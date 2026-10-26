"""Noise-injection experiment G: accuracy of each decoder on the clean
22-position montage with m added class-free noise electrodes, with and
without precision-diagonal deletion back to the clean montage size."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.analysis import load, paired, subject_level  # noqa: E402

pd.set_option("display.width", 250)
DS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["bnci2014_001", "cho2017", "lee2019_mi",
                                                       "schirrmeister2017"]
for ds in DS:
    d = load("G", ds)
    if d.empty:
        continue
    s = subject_level(d)
    print(f"\n=== G {ds}: {s.subject.nunique()} subjects")
    clean = s[(s.montage == "stationary:0") & (s.selector == "FULL")].set_index(["decoder", "subject"])["acc"]
    rows = []
    for (mont, sel, dec), g in s.groupby(["montage", "selector", "decoder"]):
        if mont == "stationary:0":
            continue
        r = paired(g.set_index("subject")["acc"], clean.loc[dec])
        ntype, m = mont.split(":")
        rows.append(dict(noise=ntype, m=int(m), selector=sel, decoder=dec, d=round(r["mean"], 1),
                         lo=round(r["lo90"], 1), hi=round(r["hi90"], 1)))
    t = pd.DataFrame(rows)
    print("clean-montage accuracy:", (100 * clean.groupby("decoder").mean()).round(1).to_dict())
    print("difference from clean montage (pp), FULL = with noise electrodes, PREC = after deletion")
    print(t.pivot_table(index=["noise", "m"], columns=["decoder", "selector"], values="d").to_string())
