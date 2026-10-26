"""Export participant-level accuracies of every experiment.

usage: python export_participant_level.py

For each run directory runs/<exp>/<dataset>/ (and runs/rescaled/CONF/<dataset>/),
accuracies are pooled over folds and averaged over random draws, subsamples and
split seeds (mech.analysis.subject_level) and written to
results/participant_level/<exp>_<dataset>.csv.gz with one row per participant
and configuration (montage or session pair, training fraction, selector,
count rule, decoder).
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.analysis import subject_level  # noqa: E402
from mech.data import PROJ  # noqa: E402

RUNS = PROJ / "runs"
OUT = PROJ / "results" / "participant_level"
EXPS = ["A", "A2", "A3", "A4", "B", "C", "D", "E", "G", "H", "W", "CONF", "CONF_CSPNORM", "BADCH", "KSEL",
        "XSESS", "HSPLIT", "GSIZE", "OOS", "XNEST", "XNESTL", "GCOR"]


def export(root, tag):
    for ddir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(glob.glob(str(ddir / "subj_*_seed_*.csv.gz")))
        frames = []
        for f in files:
            try:
                frames.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:       # excluded participants write empty files
                continue
        frames = [f for f in frames if len(f)]
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        d["exp"] = tag
        for col, val in (("montage", "native"), ("frac", 1.0), ("sub_draw", 0), ("draw", 0),
                         ("dataset", ddir.name)):
            if col not in d:
                d[col] = val
        d["montage"] = d["montage"].fillna("native")
        s = subject_level(d, by_k="k" in d and tag in ("B",))
        s["acc"] = (100 * s["acc"]).round(3)
        s.to_csv(OUT / f"{tag}_{ddir.name}.csv.gz", index=False, compression="gzip")
        print(tag, ddir.name, len(s), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for exp in EXPS:
        if (RUNS / exp).is_dir():
            export(RUNS / exp, exp)
    if (RUNS / "rescaled" / "CONF").is_dir():
        export(RUNS / "rescaled" / "CONF", "CONF_normalised")


if __name__ == "__main__":
    main()
