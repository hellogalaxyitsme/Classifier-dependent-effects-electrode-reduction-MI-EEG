"""CSP-LDA re-evaluation on the confirmatory data (documented deviation).

The frozen CSP-LDA computes log-variance features from the raw covariances
with a floor of 1e-10. Datasets stored in V^2 (Yang2025, Stieger2021 and, for
some filters, Dreyer2023 and PhysionetMI) fall below that floor. This script
re-evaluates CSP-LDA on exactly the confirmatory folds and channel subsets with
the training and test covariances divided by the mean training channel
variance (a change of units; the CSP filters are computed from trace-normalised
covariances and are unchanged). The frozen decoder is also re-run on the same
inputs (n_correct_frozen) to verify that every configuration is reproduced.

usage: python run_conf_csp.py <dataset> [--workers N]
Output: runs/CONF_CSPNORM/<dataset>/subj_<s>_seed_<seed>.csv.gz
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN = Path(__file__).resolve().parents[2] / "analysis_plans" / "confirmatory" / "code"
sys.path.insert(0, str(FROZEN))
import run_mech as rm  # noqa: E402  (frozen module, imported read-only)
from mech.data import PROJ, channel_names, cov_path, load_subject, subjects  # noqa: E402
from mech.decoders import evaluate, subset  # noqa: E402

OUT = PROJ / "runs" / "CONF_CSPNORM"
CONF = PROJ / "runs" / "CONF"


def job(args):
    dataset, subj, seed = args
    fout = OUT / dataset / f"subj_{subj}_seed_{seed}.csv.gz"
    if fout.exists():
        return subj, seed, "skip", 0.
    t0 = time.time()
    raw = np.load(cov_path(dataset, subj), allow_pickle=True)
    fs = float(raw["fs"]) if "fs" in raw.files else np.nan
    if dataset == "physionetmi" and fs != 160.:
        return subj, seed, f"excluded (fs={fs})", 0.
    sel_f = CONF / dataset / f"sel_subj_{subj}_seed_{seed}.csv.gz"
    sel = pd.read_csv(sel_f) if sel_f.exists() and sel_f.stat().st_size > 0 else pd.DataFrame()
    d = load_subject(dataset, subj, session=None)
    S_all, y_all, T = d["S"], d["y"], d["T"]
    names = channel_names(dataset)
    low = [n.lower() for n in names]
    m22 = np.array([low.index(n.lower()) for n in rm.BNCI22 if n.lower() in low])
    rows = []
    for sess in sorted(set(d["session"].tolist())):
        msk = d["session"] == sess
        S, y = S_all[msk], y_all[msk]
        if len(np.unique(y)) < 2 or np.bincount(y).min() < rm.MIN_PER_CLASS:
            continue
        C = S.shape[1]
        for fi, (tr, te) in enumerate(rm.folds(y, seed)):
            keys = (seed, subj, fi, zlib.crc32(str(sess).encode()) % 100000)
            montage = f"session:{sess}"
            c = float(np.mean(np.trace(S[tr], axis1=1, axis2=2))) / C
            base = dict(dataset=dataset, subject=subj, seed=seed, fold=fi, montage=montage)
            cfgs = [("FULL", 0, "full", C, np.arange(C))]
            fs_sel = sel[(sel.montage == montage) & (sel.fold == fi)] if len(sel) else sel
            for _, r in fs_sel.iterrows():
                if r.selector == "RAND":
                    continue
                ch = np.array([int(v) for v in str(r.channels).split()])
                cfgs.append((r.selector, 0, r.k_rule, int(r.k), ch))
            for k_rule, k in {(r.k_rule, int(r.k)) for _, r in fs_sel.iterrows()}:
                for dr in range(10):
                    ch = np.sort(rm.rng_for(*keys, k, dr).choice(C, k, replace=False))
                    cfgs.append(("RAND", dr, k_rule, k, ch))
            if 4 <= len(m22) < C:
                cfgs.append(("M22", 0, "montage", len(m22), m22))
            for name, dr, k_rule, k, ch in cfgs:
                Atr, Ate = subset(S[tr], ch), subset(S[te], ch)
                nc0, st0, _ = evaluate("CSPLDA", Atr, y[tr], Ate, y[te], T)
                nc1, st1, _ = evaluate("CSPLDA", Atr / c, y[tr], Ate / c, y[te], T)
                rows.append(dict(base, selector=name, draw=dr, k_rule=k_rule, k=k, decoder="CSPLDA",
                                 n_test=len(te), n_correct=nc1, status=st1, n_correct_frozen=nc0,
                                 status_frozen=st0))
    fout.parent.mkdir(parents=True, exist_ok=True)
    tmp = fout.parent / f".subj_{subj}_seed_{seed}.{os.getpid()}.csv.gz"
    pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(fout)
    return subj, seed, "done", time.time() - t0


def _safe(a):
    try:
        return job(a)
    except Exception:
        return a[1], a[2], "FAILED " + traceback.format_exc()[-1500:], 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = [(a.dataset, s, sd) for sd in (42, 43, 44) for s in subjects(a.dataset)]
    print(f"{time.strftime('%F %T')} CONF_CSPNORM {a.dataset}: {len(jobs)} jobs", flush=True)
    n = 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
        for subj, seed, st, dt in pool.imap_unordered(_safe, jobs):
            n += 1
            print(f"{time.strftime('%F %T')} [{n}/{len(jobs)}] subj {subj} seed {seed}: {st} "
                  f"({dt / 60:.1f} min)", flush=True)
    print(f"{time.strftime('%F %T')} CONF_CSPNORM {a.dataset} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
