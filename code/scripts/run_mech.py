"""Mechanism experiments on per-trial covariance caches.

usage: python run_mech.py <exp> <dataset> [--workers N] [--subjects 1,2] [--seeds 42,43]

Experiments (within-session, stratified 5-fold CV; every ranking, count and
projection is computed from the outer training partition only)

A  matched-count panel: at k = r* (fold-local count rule of T1) and
   at fixed k in {8, 16, 32}, every ranking and projection in mech.selectors,
   10 random subsets, and the full montage; four decoders; split seeds 42-46.
B  count curve: k on a grid from 2 to C for T1, LO, GEO, 5 random subsets
   and the PT1/PPCA projections; seed 42.
C  training-set size: stratified subsamples (25 %, 50 % of the outer training
   partition; 3 draws) with selection and count re-estimated on the subsample.
D  montage density: the dense montages restricted to the 22 BNCI2014_001
   positions and to the 19-channel 10-20 set, with selection re-run within
   the restricted montage; split seeds 42-46.
E  selection leakage: rankings and r* computed on all trials of the subject
   (training and test), as in non-nested pipelines; seed 42.

One gzip CSV per (experiment, dataset, subject, seed) is written to
runs/<exp>/<dataset>/; existing files are skipped.
"""

from __future__ import annotations

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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech import selectors as sel  # noqa: E402
from mech.data import PROJ, channel_names, load_subject, subjects  # noqa: E402
from mech.decoders import evaluate, project, subset  # noqa: E402

DECODERS = ("MDM", "MDM_OAS", "TSLR", "CSPLDA")
RUNS = PROJ / "runs"
BNCI22 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
          "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
TEN20 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
         "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
N_RAND = {"A": 10, "B": 5, "C": 5, "D": 10, "E": 0}


def rng_for(*keys):
    return np.random.default_rng([int(k) for k in keys])


def folds(y, seed):
    _, counts = np.unique(y, return_counts=True)
    n = max(2, min(5, int(counts.min())))
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


class Fold:
    """Training-partition quantities shared by all configurations of a fold."""

    def __init__(self, Str, ytr, names, seed_keys, need_bb=True):
        self.Str, self.ytr, self.C = Str, ytr, Str.shape[1]
        self.sr = sel.ScoreRanker(Str, ytr)
        self.r_star = self.sr.rule["r_star"]
        perm = rng_for(*seed_keys, 7).permutation(ytr)
        self.sr_sh = sel.ScoreRanker(Str, perm)
        self.cspf = sel.csp_filter_ranking(Str, ytr)
        self.cspp = sel.csp_pattern_ranking(Str, ytr)
        self.bb = sel.bb_ranking(Str, ytr) if need_bb else None
        self.geo = sel.geo_ranking(names) if names is not None else None
        self.seed_keys = seed_keys

    def configs(self, k, kinds, n_rand):
        """Yield (selector, draw, kind, channels_or_projection)."""
        C = self.C
        for kind in kinds:
            if kind == "T1":
                yield kind, 0, "ch", self.sr.channels("standard", k)
            elif kind == "JG":
                yield kind, 0, "ch", self.sr.channels("jensen", k)
            elif kind == "LO":
                yield kind, 0, "ch", self.sr.channels("loading", k)
            elif kind == "SH":
                yield kind, 0, "ch", self.sr_sh.channels("standard", k)
            elif kind == "CSPF":
                yield kind, 0, "ch", self.cspf[:k]
            elif kind == "CSPP":
                yield kind, 0, "ch", self.cspp[:k]
            elif kind == "BB" and self.bb is not None:
                yield kind, 0, "ch", self.bb[:k]
            elif kind == "GEO" and self.geo is not None:
                yield kind, 0, "ch", self.geo[:k]
            elif kind == "NS":
                lv = np.log(np.diagonal(self.Str, axis1=1, axis2=2))
                yield kind, 0, "ch", np.argsort(lv.std(0), kind="stable")[:k]
            elif kind == "VAR":
                lv = np.log(np.diagonal(self.Str, axis1=1, axis2=2))
                yield kind, 0, "ch", np.argsort(lv.mean(0), kind="stable")[:k]
            elif kind in ("PREC", "VIF"):
                mix = self.Str.mean(0)
                P = np.linalg.inv(mix + 1e-4 * np.trace(mix) / C * np.eye(C))
                score = np.diag(P) if kind == "PREC" else np.diag(mix) * np.diag(P)
                yield kind, 0, "ch", np.argsort(-score, kind="stable")[:k]
            elif kind == "PT1":
                yield kind, 0, "proj", self.sr.projection("standard", k)
            elif kind == "PJG":
                yield kind, 0, "proj", self.sr.projection("jensen", k)
            elif kind == "PPCA":
                yield kind, 0, "proj", sel.pca_projection(self.Str, k)
        for d in range(n_rand):
            yield "RAND", d, "ch", np.sort(rng_for(*self.seed_keys, k, d).choice(C, k, replace=False))


def run_configs(fold, Ste, yte, T, ks, kinds, n_rand, decoders, base, rows, sel_rows=None):
    """Evaluate the full montage and every (k, selector) configuration."""
    for dec in decoders:
        nc, st, t = evaluate(dec, fold.Str, fold.ytr, Ste, yte, T)
        rows.append(dict(base, selector="FULL", draw=0, k_rule="full", k=fold.C, decoder=dec,
                         n_test=len(yte), n_correct=nc, status=st, fit_s=round(t, 3)))
    for k_rule, k in ks:
        if not (1 <= k < fold.C):
            continue
        for name, d, kind, obj in fold.configs(k, kinds, n_rand):
            if kind == "ch":
                A_tr, A_te = subset(fold.Str, obj), subset(Ste, obj)
                if sel_rows is not None and d == 0:
                    sel_rows.append(dict(base, selector=name, k_rule=k_rule, k=k,
                                         channels=" ".join(map(str, np.sort(obj)))))
            else:
                A_tr, A_te = project(fold.Str, obj), project(Ste, obj)
            for dec in decoders:
                if name == "RAND" and dec == "TSLR" and d >= 5:
                    continue
                nc, st, t = evaluate(dec, A_tr, fold.ytr, A_te, yte, T)
                rows.append(dict(base, selector=name, draw=d, k_rule=k_rule, k=k, decoder=dec,
                                 n_test=len(yte), n_correct=nc, status=st, fit_s=round(t, 3)))


KINDS_A = ["T1", "JG", "LO", "SH", "CSPF", "CSPP", "BB", "GEO", "PT1", "PJG", "PPCA"]
KINDS_A2 = ["NS", "VAR"]
KINDS_A3 = ["PREC", "VIF"]
KINDS_A4 = ["T1", "PREC", "SH", "BB", "CSPF", "GEO"]
KINDS_B = ["T1", "LO", "GEO", "PT1", "PPCA"]
KINDS_C = ["T1", "PT1"]
KINDS_D = ["T1", "JG", "LO", "CSPF", "BB", "PT1"]
KINDS_E = ["T1", "CSPF", "BB"]
GRID = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]


def job(args):
    exp, dataset, subj, seed = args
    out_dir = RUNS / exp / dataset
    fout = out_dir / f"subj_{subj}_seed_{seed}.csv.gz"
    if fout.exists():
        return subj, seed, "skip", 0.
    t0 = time.time()
    if exp == "CONF":
        return conf_job(dataset, subj, seed, fout, t0)
    d = load_subject(dataset, subj)
    S, y, T = d["S"], d["y"], d["T"]
    try:
        names = channel_names(dataset)
    except FileNotFoundError:
        names = None
    rows, sel_rows, fold_rows = [], [], []
    for fi, (tr, te) in enumerate(folds(y, seed)):
        keys = (seed, subj, fi)
        if exp == "G":
            # causal test: independent, class-free noise electrodes added to a clean montage
            low = [n.lower() for n in names]
            idx = np.array([low.index(n.lower()) for n in BNCI22 if n.lower() in low])
            S0 = subset(S, idx)
            C0 = len(idx)
            scale = float(np.median(np.diagonal(S0[tr].mean(0))))
            for ntype, sig in (("stationary", 0.0), ("nonstat05", 0.5), ("nonstat10", 1.0)):
                for m in (0, 4, 8, 16, 32):
                    if m == 0 and ntype != "stationary":
                        continue
                    rs = rng_for(seed, subj, m, int(sig * 10), 99)
                    n = len(y)
                    if sig == 0:
                        v = scale * rs.chisquare(T, size=(n, m)) / T
                    else:
                        v = scale * np.exp(sig * rs.standard_normal((n, m)) - sig ** 2 / 2)
                    Sa = np.zeros((n, C0 + m, C0 + m))
                    Sa[:, :C0, :C0] = S0
                    Sa[:, np.arange(C0, C0 + m), np.arange(C0, C0 + m)] = v
                    fold = Fold(Sa[tr], y[tr], None, keys, need_bb=False)
                    base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                                sub_draw=0, montage=f"{ntype}:{m}", r_star=fold.r_star)
                    ks = [("fixed", C0)] if m > 0 else []
                    run_configs(fold, Sa[te], y[te], T, ks, ["PREC"], 0, ("MDM", "TSLR", "FGMDM", "CSPLDA"),
                                base, rows)
                    if m > 0:
                        keep = np.argsort(-np.diag(np.linalg.inv(
                            Sa[tr].mean(0) + 1e-4 * np.trace(Sa[tr].mean(0)) / (C0 + m) * np.eye(C0 + m))))[:C0]
                        fold_rows.append(dict(base, frac_kept_real=float(np.mean(keep < C0))))
        elif exp in ("H", "W"):
            # H: common-average reference (one channel dropped to keep full rank); W: shorter windows
            if exp == "H":
                C = S.shape[1]
                Hm = np.eye(C) - 1. / C
                Sx = (Hm @ S @ Hm)[:, :C - 1, :C - 1]
                nx = names[:C - 1]
            else:
                Sx, nx = S, names
            fold = Fold(Sx[tr], y[tr], nx, keys, need_bb=False)
            base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                        sub_draw=0, montage="car" if exp == "H" else f"T{T}", r_star=fold.r_star)
            ks = [("rstar", fold.r_star), ("fixed", 16)]
            run_configs(fold, Sx[te], y[te], T, ks, ["T1", "PREC", "SH", "GEO"], 5,
                        ("MDM", "TSLR", "FGMDM", "CSPLDA"), base, rows)
        elif exp == "A4":
            fold = Fold(S[tr], y[tr], names, keys, need_bb=True)
            base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                        sub_draw=0, montage="native", r_star=fold.r_star)
            ks = [("rstar", fold.r_star), ("fixed", 16)]
            run_configs(fold, S[te], y[te], T, ks, KINDS_A4, 5, ("FGMDM",), base, rows)
        elif exp in ("A2", "A3"):
            fold = Fold(S[tr], y[tr], names, keys, need_bb=False)
            base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                        sub_draw=0, montage="native", r_star=fold.r_star)
            ks = [("rstar", fold.r_star)] + [("fixed", k) for k in (8, 16, 32)]
            run_configs(fold, S[te], y[te], T, ks, KINDS_A2 if exp == "A2" else KINDS_A3, 0, DECODERS,
                        base, rows, sel_rows)
        elif exp in ("A", "B"):
            fold = Fold(S[tr], y[tr], names, keys, need_bb=(exp == "A"))
            base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                        sub_draw=0, montage="native", r_star=fold.r_star)
            if exp == "A":
                ks = [("rstar", fold.r_star)] + [("fixed", k) for k in (8, 16, 32)]
                run_configs(fold, S[te], y[te], T, ks, KINDS_A, N_RAND["A"], DECODERS, base, rows,
                            sel_rows)
            else:
                ks = [("grid", k) for k in GRID]
                run_configs(fold, S[te], y[te], T, ks, KINDS_B, N_RAND["B"], DECODERS, base, rows)
            fold_rows.append(dict(base, r_rel=fold.sr.rule["r_rel"], r_pr=fold.sr.rule["r_pr"],
                                  loading_share=fold.sr.loading_share, n_train=len(tr), n_test=len(te)))
        elif exp == "C":
            for frac in (0.25, 0.5):
                for sd in range(3):
                    rs = rng_for(seed, subj, fi, int(frac * 100), sd)
                    sub = np.concatenate([rs.choice(tr[y[tr] == c], max(2, int(round(frac * np.sum(y[tr] == c)))),
                                                    replace=False) for c in np.unique(y)])
                    fold = Fold(S[sub], y[sub], names, (seed, subj, fi, sd, int(frac * 100)), need_bb=False)
                    base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=frac,
                                sub_draw=sd, montage="native", r_star=fold.r_star)
                    ks = [("rstar", fold.r_star), ("fixed", 16)]
                    run_configs(fold, S[te], y[te], T, ks, KINDS_C, N_RAND["C"], DECODERS, base, rows)
                    fold_rows.append(dict(base, r_rel=fold.sr.rule["r_rel"], r_pr=fold.sr.rule["r_pr"],
                                          loading_share=fold.sr.loading_share, n_train=len(sub),
                                          n_test=len(te)))
        elif exp == "D":
            for mname, mlist in (("bnci22", BNCI22), ("ten20", TEN20)):
                low = [n.lower() for n in names]
                idx = np.array([low.index(n.lower()) for n in mlist if n.lower() in low])
                Sm = subset(S, idx)
                mnames = [names[i] for i in idx]
                fold = Fold(Sm[tr], y[tr], mnames, keys + (len(idx),), need_bb=True)
                base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                            sub_draw=0, montage=f"{mname}:{len(idx)}", r_star=fold.r_star)
                run_configs(fold, Sm[te], y[te], T, [("rstar", fold.r_star)], KINDS_D, N_RAND["D"],
                            DECODERS, base, rows)
                fold_rows.append(dict(base, r_rel=fold.sr.rule["r_rel"], r_pr=fold.sr.rule["r_pr"],
                                      loading_share=fold.sr.loading_share, n_train=len(tr), n_test=len(te)))
        elif exp == "E":
            allf = Fold(S, y, names, keys, need_bb=True)        # selection sees every trial
            fold = Fold(S[tr], y[tr], names, keys, need_bb=False)
            fold.sr, fold.cspf, fold.bb, fold.r_star = allf.sr, allf.cspf, allf.bb, allf.r_star
            base = dict(exp=exp, dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                        sub_draw=0, montage="native", r_star=fold.r_star)
            run_configs(fold, S[te], y[te], T, [("rstar", fold.r_star)], KINDS_E, 0,
                        ("MDM", "MDM_OAS", "TSLR", "CSPLDA"), base, rows)
            fold_rows.append(dict(base, n_train=len(tr), n_test=len(te)))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".subj_{subj}_seed_{seed}.{os.getpid()}.csv.gz"
    pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
    if sel_rows:
        pd.DataFrame(sel_rows).to_csv(out_dir / f"sel_subj_{subj}_seed_{seed}.csv.gz", index=False,
                                      compression="gzip")
    pd.DataFrame(fold_rows).to_csv(out_dir / f"folds_subj_{subj}_seed_{seed}.csv.gz", index=False,
                                   compression="gzip")
    for f in out_dir.glob(f"*subj_{subj}_seed_{seed}*.csv.gz"):
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())
    tmp.replace(fout)
    return subj, seed, "done", time.time() - t0


KINDS_CONF = ["T1", "SH", "PREC", "BB", "CSPF", "GEO"]
DEC_CONF = ("MDM", "MDM_OAS", "FGMDM", "TSLR", "CSPLDA")
MIN_PER_CLASS = 20


def conf_job(dataset, subj, seed, fout, t0):
    """Frozen confirmatory protocol: within-session 5-fold CV in every recording
    session with at least MIN_PER_CLASS trials per class."""
    from mech.data import cov_path
    raw = np.load(cov_path(dataset, subj), allow_pickle=True)
    fs = float(raw["fs"]) if "fs" in raw.files else np.nan
    if dataset == "physionetmi" and fs != 160.:
        return subj, seed, f"excluded (fs={fs})", 0.     # prespecified: recordings at 128 Hz
    d = load_subject(dataset, subj, session=None)
    S_all, y_all, T = d["S"], d["y"], d["T"]
    names = channel_names(dataset)
    low = [n.lower() for n in names]
    m22 = np.array([low.index(n.lower()) for n in BNCI22 if n.lower() in low])
    rows, sel_rows, fold_rows = [], [], []
    out_dir = fout.parent
    for sess in sorted(set(d["session"].tolist())):
        msk = d["session"] == sess
        S, y = S_all[msk], y_all[msk]
        if len(np.unique(y)) < 2 or np.bincount(y).min() < MIN_PER_CLASS:
            fold_rows.append(dict(exp="CONF", dataset=dataset, subject=subj, seed=seed, session=sess,
                                  skipped="too_few_trials", n=len(y)))
            continue
        for fi, (tr, te) in enumerate(folds(y, seed)):
            keys = (seed, subj, fi, zlib.crc32(str(sess).encode()) % 100000)
            fold = Fold(S[tr], y[tr], names, keys, need_bb=True)
            base = dict(exp="CONF", dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0,
                        sub_draw=0, montage=f"session:{sess}", r_star=fold.r_star)
            ks = [("rstar", fold.r_star), ("fixed", 16)]
            run_configs(fold, S[te], y[te], T, ks, KINDS_CONF, 10, DEC_CONF, base, rows, sel_rows)
            if 4 <= len(m22) < S.shape[1]:
                for dec in DEC_CONF:
                    nc, st, t = evaluate(dec, subset(S[tr], m22), y[tr], subset(S[te], m22), y[te], T)
                    rows.append(dict(base, selector="M22", draw=0, k_rule="montage", k=len(m22), decoder=dec,
                                     n_test=len(te), n_correct=nc, status=st, fit_s=round(t, 3)))
            fold_rows.append(dict(base, session=sess, r_rel=fold.sr.rule["r_rel"], r_pr=fold.sr.rule["r_pr"],
                                  loading_share=fold.sr.loading_share, n_train=len(tr), n_test=len(te), fs=fs))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("sel_", sel_rows), ("folds_", fold_rows)):
        pd.DataFrame(data).to_csv(out_dir / f"{name}subj_{subj}_seed_{seed}.csv.gz", index=False,
                                  compression="gzip")
    tmp = out_dir / f".subj_{subj}_seed_{seed}.{os.getpid()}.csv.gz"
    pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(fout)
    return subj, seed, "done", time.time() - t0


def _safe(a):
    try:
        return job(a)
    except Exception:
        return a[2], a[3], "FAILED " + traceback.format_exc()[-1500:], 0.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["A", "A2", "A3", "A4", "B", "C", "D", "E", "G", "H", "W", "CONF"])
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    seeds = ([int(s) for s in a.seeds.split(",")] if a.seeds else
             ([42, 43, 44, 45, 46] if a.exp in ("A", "A2", "A3", "A4", "D") else
              [42, 43, 44] if a.exp == "CONF" else [42]))
    subs = subjects(a.dataset) if a.subjects == "all" else [int(s) for s in a.subjects.split(",")]
    jobs = [(a.exp, a.dataset, s, sd) for sd in seeds for s in subs]
    print(f"{time.strftime('%F %T')} exp {a.exp} {a.dataset}: {len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    n = 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
        for subj, seed, st, dt in pool.imap_unordered(_safe, jobs):
            n += 1
            print(f"{time.strftime('%F %T')} [{n}/{len(jobs)}] subj {subj} seed {seed}: {st} "
                  f"({dt / 60:.1f} min; elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
    print(f"{time.strftime('%F %T')} exp {a.exp} {a.dataset} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
