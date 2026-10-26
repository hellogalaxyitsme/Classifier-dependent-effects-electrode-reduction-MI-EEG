"""Two follow-up experiments on the cached covariances.

usage: python run_transfer_mixing.py <exp> <dataset> [--workers N] [--seeds 42]

XNEST  cross-session transfer of the label-free precision ranking (PREC) with
       its channel count chosen by inner 5-fold cross-validation on the
       training session (separately for MDM and TS-LR, grid as in KSEL) and
       then frozen for the target session; evaluated without adaptation and
       after unsupervised re-centring of each session, alongside the full
       montage and M22
XNESTL as XNEST, with the diagonal loading of MDM added before re-centring, so
       that sessions with numerically rank-deficient trial covariances can be
       re-centred (sensitivity analysis)
GCOR  covariance-block augmentation with correlated extra electrodes: 32
       class-independent electrodes (trial-varying, sigma = 1, or stationary)
       are appended to the 22-electrode montage as in GSIZE, and the sources
       are mixed by a fixed matrix L = [[I, G], [B, I]], so that the extra
       electrodes share activity with the informative ones (B) and, in the
       'leak' condition, the informative electrodes also receive part of the
       nuisance activity (G)

Protocols, folds and normalisation follow run_gaps.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gaps as rg  # noqa: E402
import run_mech as rm  # noqa: E402
from mech.data import channel_names, load_subject  # noqa: E402
from mech.decoders import evaluate, fit_predict, subset  # noqa: E402

N_EXTRA = 32
MIXES = (("indep", 0.0, 0.0), ("spread", 1.0, 0.0), ("leak", 1.0, np.sqrt(0.1)))


def job_xnest(dataset, subj, seed, rows, sel_rows, fold_rows):
    names = channel_names(dataset)
    m22 = rg.m22_idx(names)
    d = load_subject(dataset, subj, session=None)
    S_all, y_all, T = d["S"], d["y"], d["T"]
    lab = d["session"]
    for a, b in rg.xsess_pairs(dataset, d):
        tr, te = np.flatnonzero(lab == a), np.flatnonzero(lab == b)
        C = S_all.shape[1]
        c = np.trace(S_all[tr], axis1=1, axis2=2).mean() / C
        Str, Ste, ytr, yte = S_all[tr] / c, S_all[te] / c, y_all[tr], y_all[te]
        K = len(np.unique(ytr))
        grid = sorted({k for k in rg.GRID if max(K, 2) <= k < C} | {C})
        acc = {dec: np.zeros(len(grid)) for dec in ("MDM", "TSLR")}
        inner = StratifiedKFold(n_splits=max(2, min(5, int(np.bincount(ytr).min()))), shuffle=True,
                                random_state=seed + 1000)
        for itr, iva in inner.split(np.zeros(len(ytr)), ytr):
            A, B, ya, yb = Str[itr], Str[iva], ytr[itr], ytr[iva]
            order = rg.prec_rank(A)
            for gi, k in enumerate(grid):
                ch = np.sort(order[:k]) if k < C else np.arange(C)
                for dec in ("MDM", "TSLR"):
                    try:
                        yp = (fit_predict("MDM", subset(A, ch), ya, subset(B, ch), T) if dec == "MDM" else
                              rg.tslr_fixed(subset(A, ch), ya, subset(B, ch), T))
                        acc[dec][gi] += np.mean(yp == yb)
                    except Exception:
                        pass
        prec = rg.prec_rank(Str)
        kb = {dec: grid[int(np.argmax(acc[dec]))] for dec in ("MDM", "TSLR")}
        base = dict(exp="XNEST", dataset=dataset, subject=subj, seed=seed, fold=0, frac=1.0, sub_draw=0,
                    montage=f"{a}->{b}")
        for tf, tag in ((None, ""), (rg.recenter, ":rc")):
            b2 = dict(base, montage=base["montage"] + tag)
            full = rg.eval_full(("MDM", "TSLR"), Str, ytr, Ste, yte, T, b2, rows, tf)
            for dec in ("MDM", "TSLR"):
                ch = prec[:kb[dec]] if kb[dec] < C else np.arange(C)
                rg.run_cfgs([("PREC", 0, "innercv", ch)], (dec,), Str, ytr, Ste, yte, T, b2, rows,
                            sel_rows if tf is None else None, full, tf)
            if 4 <= len(m22) < C:
                rg.run_cfgs([("M22", 0, "montage", m22)], ("MDM", "TSLR"), Str, ytr, Ste, yte, T, b2, rows,
                            None, full, tf)
        fold_rows.append(dict(base, k_mdm=kb["MDM"], k_tslr=kb["TSLR"], C=C, n_train=len(tr), n_test=len(te)))


def recenter_loaded(X):
    """Re-centring after the diagonal loading used by MDM (1e-6 of the mean
    channel variance of each trial), which keeps numerically rank-deficient
    trial covariances positive definite."""
    C = X.shape[1]
    load = 1e-6 * np.trace(X, axis1=1, axis2=2)[:, None, None] / C * np.eye(C)
    return _RECENTER(X + load)


_RECENTER = rg.recenter


def job_xnestl(dataset, subj, seed, rows, sel_rows, fold_rows):
    """XNEST with loaded re-centring (sensitivity analysis for the participants
    whose unloaded session means could not be computed)."""
    rg.recenter = recenter_loaded
    try:
        job_xnest(dataset, subj, seed, rows, sel_rows, fold_rows)
    finally:
        rg.recenter = _RECENTER


def job_gcor(dataset, subj, seed, rows, sel_rows, fold_rows):
    d = load_subject(dataset, subj)
    S, y, T = d["S"], d["y"], d["T"]
    idx = rg.m22_idx(channel_names(dataset))
    S0 = subset(S, idx)
    C0, n, m = len(idx), len(y), N_EXTRA
    rmix = rm.rng_for(seed, subj, 555)
    G_spread = rmix.standard_normal((m, C0)) / np.sqrt(C0)
    G_leak = rmix.standard_normal((C0, m)) / np.sqrt(m)
    for fi, (tr, te) in enumerate(rm.folds(y, seed)):
        base = dict(exp="GCOR", dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0, sub_draw=0)
        c0 = np.trace(S0[tr], axis1=1, axis2=2).mean() / C0
        for dec in ("MDM", "TSLR", "TSLR_RAW"):
            nc, st, _ = evaluate(dec, S0[tr] / c0, y[tr], S0[te] / c0, y[te], T)
            rows.append(dict(base, montage="clean", selector="FULL", draw=0, k_rule="full", k=C0, decoder=dec,
                             n_test=len(te), n_correct=nc, status=st))
        scale = float(np.median(np.diagonal(S0[tr].mean(0))))
        for ntype, sig in (("stationary", 0.0), ("nonstat10", 1.0)):
            rs = rm.rng_for(seed, subj, m, int(sig * 10), 99)
            v = (scale * rs.chisquare(T, size=(n, m)) / T if sig == 0 else
                 scale * np.exp(sig * rs.standard_normal((n, m)) - sig ** 2 / 2))
            Sa = np.zeros((n, C0 + m, C0 + m))
            Sa[:, :C0, :C0] = S0
            Sa[:, np.arange(C0, C0 + m), np.arange(C0, C0 + m)] = v
            for mix, bsp, glk in MIXES:
                L = np.eye(C0 + m)
                L[C0:, :C0] = bsp * G_spread
                L[:C0, C0:] = glk * G_leak
                X = L @ Sa @ L.T
                c = np.trace(X[tr], axis1=1, axis2=2).mean() / (C0 + m)
                Xtr, Xte = X[tr] / c, X[te] / c
                b2 = dict(base, montage=f"{ntype}:{mix}")
                for dec in ("MDM", "TSLR", "TSLR_RAW"):
                    nc, st, _ = evaluate(dec, Xtr, y[tr], Xte, y[te], T)
                    rows.append(dict(b2, selector="FULL", draw=0, k_rule="full", k=C0 + m, decoder=dec,
                                     n_test=len(te), n_correct=nc, status=st))
                keep = np.sort(rg.prec_rank(Xtr)[:C0])
                for name, ch in (("PREC", keep), ("ORIG", np.arange(C0))):
                    nc, st, _ = evaluate("MDM", subset(Xtr, ch), y[tr], subset(Xte, ch), y[te], T)
                    rows.append(dict(b2, selector=name, draw=0, k_rule="fixed", k=C0, decoder="MDM",
                                     n_test=len(te), n_correct=nc, status=st))
                fold_rows.append(dict(b2, frac_kept_orig=float(np.mean(keep < C0))))


rg.JOBS["XNEST"] = job_xnest
rg.JOBS["XNESTL"] = job_xnestl
rg.JOBS["GCOR"] = job_gcor

if __name__ == "__main__":
    rg.main()
