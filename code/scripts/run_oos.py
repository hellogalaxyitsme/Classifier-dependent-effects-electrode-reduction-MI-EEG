"""Out-of-sample confirmation of the nested channel-count and fixed-montage
results (PREREG_OOS.md).

usage: python run_oos.py OOS <dataset> [--workers N] [--seeds 42,43,44] [--subjects 1,2]

The protocol is the KSEL protocol of run_gaps.py, unchanged, applied to the
first session with at least 20 trials per class, plus the fixed 22-position
central montage (M22) for MDM and TS-LR when at least 18 of its positions are
present.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gaps as rg  # noqa: E402
import run_mech as rm  # noqa: E402
from mech.data import channel_names  # noqa: E402

_within = rg.within_units


def first_session(dataset, subj, max_sessions=None):
    return _within(dataset, subj, max_sessions=1)


rg.within_units = first_session          # module-level, so spawned workers inherit it


def job_oos(dataset, subj, seed, rows, sel_rows, fold_rows):
    n0, f0 = len(rows), len(fold_rows)
    rg.job_ksel(dataset, subj, seed, rows, sel_rows, fold_rows)
    for r in rows[n0:] + fold_rows[f0:]:
        r["exp"] = "OOS"
    m22 = rg.m22_idx(channel_names(dataset))
    for sess, S, y, T in rg.within_units(dataset, subj):
        C = S.shape[1]
        if not 18 <= len(m22) < C:
            continue
        for fi, (tr, te) in enumerate(rm.folds(y, seed)):
            c = np.trace(S[tr], axis1=1, axis2=2).mean() / C
            Str, Ste, ytr, yte = S[tr] / c, S[te] / c, y[tr], y[te]
            base = dict(exp="OOS", dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0, sub_draw=0,
                        montage=f"session:{sess}")
            rg.run_cfgs([("M22", 0, "montage", m22)], ("MDM", "TSLR"), Str, ytr, Ste, yte, T, base, rows,
                        sel_rows)


rg.JOBS["OOS"] = job_oos

if __name__ == "__main__":
    rg.main()
