"""Exploratory follow-up experiments, run after the confirmatory analysis.

usage: python run_gaps.py <exp> <dataset> [--workers N] [--seeds 42,43,44] [--subjects 1,2]

BADCH   standard bad-channel detection (PREP-style deviation, correlation and
        RANSAC-predictability criteria computed from the training
        covariances) against the label-free precision ranking (PREC), the
        variance-fluctuation ranking (NS), the geometric ranking (GEO) and
        random subsets at the same channel count
KSEL    choice of the channel count: k chosen by inner 5-fold CV on the outer
        training partition (separately for MDM and for TS-LR) and fixed
        fractions of the montage, for the PREC and GEO rankings
XSESS   cross-session transfer: selection and decoders fitted on one session,
        evaluated on another session of the same subject
HSPLIT  Schirrmeister2017 (HGD) with its original training and test runs
GSIZE   noise-electrode injection (as experiment G of run_mech.py) crossed
        with the size of the training partition

Protocols follow run_mech.py: within-session stratified 5-fold CV, sessions
with at least 20 trials per class (development datasets: first session, as
in the development experiments), every ranking, count, detector and decoder
fitted on training trials only. All covariances of a fold are divided by the
mean training channel variance, a change of units that leaves MDM, TS-LR and
FgMDM unchanged and removes the clipping floor of the original CSP-LDA
features (see run_conf_csp.py).
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
import run_mech as rm  # noqa: E402
from mech import selectors as sel  # noqa: E402
from mech.data import PROJ, channel_names, cov_path, load_subject, subjects  # noqa: E402
from mech.decoders import evaluate, fit_predict, oas, subset  # noqa: E402

RUNS = PROJ / "runs"
DEVSETS = {"cho2017", "lee2019_mi", "schirrmeister2017", "bnci2014_001"}
GRID = (4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96)
PAIRS = {"bnci2014_001": [("0train", "1test"), ("1test", "0train")],
         "lee2019_2s": [("1", "2"), ("2", "1")],
         "yang2025": [("0", "1"), ("1", "2"), ("0", "2")]}


def sesskey(s):
    try:
        return (0, int(s), "")
    except ValueError:
        return (1, 0, str(s))


def decoders_for(dataset, exp):
    dec = ["MDM", "TSLR", "FGMDM", "CSPLDA"]
    if dataset == "schirrmeister2017" and exp in ("BADCH",):
        dec.remove("FGMDM")                 # 128-channel FgMDM fits are prohibitively slow
    return tuple(dec)


# ---------------------------------------------------------------------------
# data units
# ---------------------------------------------------------------------------
def within_units(dataset, subj, max_sessions=None):
    """(session, S, y, T) for every session analysed within-session."""
    if dataset in DEVSETS:
        d = load_subject(dataset, subj)
        return [("first", d["S"], d["y"], d["T"])]
    raw = np.load(cov_path(dataset, subj), allow_pickle=True)
    if dataset == "physionetmi" and float(raw["fs"]) != 160.:
        return []
    d = load_subject(dataset, subj, session=None)
    out = []
    for sess in sorted(set(d["session"].tolist()), key=sesskey):
        m = d["session"] == sess
        y = d["y"][m]
        if len(np.unique(y)) < 2 or np.bincount(y).min() < rm.MIN_PER_CLASS:
            continue
        out.append((sess, d["S"][m], y, d["T"]))
    return out[:max_sessions] if max_sessions else out


# ---------------------------------------------------------------------------
# rankings and detectors (training covariances only)
# ---------------------------------------------------------------------------
def prec_rank(Str):
    C = Str.shape[1]
    mix = Str.mean(0)
    P = np.linalg.inv(mix + 1e-4 * np.trace(mix) / C * np.eye(C))
    return np.argsort(-np.diag(P), kind="stable")


def ns_rank(Str):
    lv = np.log(np.diagonal(Str, axis1=1, axis2=2))
    return np.argsort(lv.std(0), kind="stable")


def sphere_positions(names):
    import mne
    mont = mne.channels.make_standard_montage("standard_1005")
    ref = {k.lower(): v for k, v in mont.get_positions()["ch_pos"].items()}
    pos = np.full((len(names), 3), np.nan)
    for i, n in enumerate(names):
        if n.lower() in ref:
            pos[i] = ref[n.lower()]
    ok = ~np.isnan(pos[:, 0])
    P = pos[ok]
    sol = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]
    pos = pos - sol[:3]
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    return pos, ok


def ransac_weights(pos, ok, rs, n_sub=50, frac=0.25):
    """Average spherical-spline prediction weights of every channel from random
    subsets of 25 % of the channels that exclude it (PREP RANSAC defaults; the
    mean rather than the median of the subset predictions, so that the
    prediction is linear and its correlation follows from the covariance)."""
    from mne.channels.interpolation import _make_interpolation_matrix
    C = len(pos)
    good = np.flatnonzero(ok)
    m = max(4, int(np.ceil(frac * len(good))))
    Wsum, cnt = np.zeros((C, C)), np.zeros(C)
    for _ in range(n_sub):
        sub = np.sort(rs.choice(good, m, replace=False))
        M = _make_interpolation_matrix(pos[sub], pos[good])
        Wf = np.zeros((C, C))
        Wf[np.ix_(good, sub)] = M
        out = np.setdiff1d(good, sub)
        Wsum[out] += Wf[out]
        cnt[out] += 1
    return Wsum / np.maximum(cnt, 1)[:, None], cnt > 0


def detect(Str, W, wok):
    """PREP-style bad-channel criteria (single pass, no robust re-referencing).
    Each trial is one window."""
    n, C, _ = Str.shape
    d = np.diagonal(Str, axis1=1, axis2=2)
    amp = np.sqrt(d.mean(0))                                            # deviation
    q75, q25 = np.percentile(amp, [75, 25])
    z = (amp - np.median(amp)) / max(0.7413 * (q75 - q25), 1e-30)
    bad_dev = np.abs(z) > 5.
    R = Str / np.sqrt(d[:, :, None] * d[:, None, :])                    # correlation
    R[:, np.arange(C), np.arange(C)] = 0.
    bad_cor = (np.abs(R).max(2) < 0.4).mean(0) > 0.01
    num = np.einsum("kj,tjk->tk", W, Str)                               # RANSAC predictability
    den = np.einsum("kj,tjl,kl->tk", W, Str, W)
    corr = num / np.sqrt(np.maximum(den * d, 1e-300))
    bad_ran = ((corr < 0.75).mean(0) > 0.4) & wok
    return dict(PREP_DEV=bad_dev, PREP_COR=bad_cor, PREP_RAN=bad_ran,
                PREP_ALL=bad_dev | bad_cor | bad_ran)


def tslr_fixed(Str, ytr, Ste, T):
    """TS-LR with a fixed regularisation strength (C = 1), used only inside the
    inner count search to bound its cost."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ts = TangentSpace(metric="riemann")
    F = ts.fit_transform(oas(Str, T))
    sc = StandardScaler().fit(F)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(F), ytr)
    return clf.predict(sc.transform(ts.transform(oas(Ste, T))))


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def recenter(X):
    """Unsupervised re-centring of one session (Zanini et al. 2018): congruence
    by the inverse square root of the session's Riemannian mean."""
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann
    Mi = invsqrtm(mean_riemann(X))
    return Mi @ X @ Mi


def run_cfgs(cfgs, decs, Str, ytr, Ste, yte, T, base, rows, sel_rows=None, full=None, tf=None):
    C = Str.shape[1]
    for name, dr, k_rule, ch in cfgs:
        ch = np.sort(np.asarray(ch, dtype=int))
        if len(ch) < 2:
            continue
        if sel_rows is not None and dr == 0:
            sel_rows.append(dict(base, selector=name, k_rule=k_rule, k=len(ch), channels=" ".join(map(str, ch))))
        A, B = None, None
        for dec in decs:
            if name == "RAND" and dec == "TSLR" and dr >= 5:
                continue
            if len(ch) == C and full is not None:
                nc, st = full[dec]
            else:
                if A is None:
                    A, B = subset(Str, ch), subset(Ste, ch)
                    if tf is not None:
                        A, B = tf(A), tf(B)
                nc, st, _ = evaluate(dec, A, ytr, B, yte, T)
            rows.append(dict(base, selector=name, draw=dr, k_rule=k_rule, k=len(ch), decoder=dec,
                             n_test=len(yte), n_correct=nc, status=st))


def eval_full(decs, Str, ytr, Ste, yte, T, base, rows, tf=None):
    full = {}
    if tf is not None:
        Str, Ste = tf(Str), tf(Ste)
    for dec in decs:
        nc, st, _ = evaluate(dec, Str, ytr, Ste, yte, T)
        full[dec] = (nc, st)
        rows.append(dict(base, selector="FULL", draw=0, k_rule="full", k=Str.shape[1], decoder=dec,
                         n_test=len(yte), n_correct=nc, status=st))
    return full


def m22_idx(names):
    low = [n.lower() for n in names]
    return np.array([low.index(n.lower()) for n in rm.BNCI22 if n.lower() in low])


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def job_badch(dataset, subj, seed, rows, sel_rows, fold_rows):
    names = channel_names(dataset)
    pos, ok = sphere_positions(names)
    W, wok = ransac_weights(pos, ok, rm.rng_for(subj, 77))
    geo = sel.geo_ranking(names)
    decs = decoders_for(dataset, "BADCH")
    for sess, S, y, T in within_units(dataset, subj, max_sessions=3 if dataset == "stieger2021" else None):
        C, K = S.shape[1], len(np.unique(y))
        for fi, (tr, te) in enumerate(rm.folds(y, seed)):
            keys = (seed, subj, fi, zlib.crc32(str(sess).encode()) % 100000)
            c = np.trace(S[tr], axis1=1, axis2=2).mean() / C
            Str, Ste, ytr, yte = S[tr] / c, S[te] / c, y[tr], y[te]
            base = dict(exp="BADCH", dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0, sub_draw=0,
                        montage=f"session:{sess}")
            full = eval_full(decs, Str, ytr, Ste, yte, T, base, rows)
            bad = detect(Str, W, wok)
            prec, ns = prec_rank(Str), ns_rank(Str)
            sr = sel.ScoreRanker(Str, ytr)
            r_star = sr.rule["r_star"]
            cfgs = [(crit, 0, "detect", np.flatnonzero(~b)) for crit, b in bad.items()]
            k = int(C - bad["PREP_ALL"].sum())
            if k == C:                      # nothing flagged: every matched subset is the full montage
                cfgs += [(nm, 0, "matched", np.arange(C)) for nm in ("PREC", "NS", "GEO")]
                cfgs += [("RAND", d, "matched", np.arange(C)) for d in range(5)]
            elif k >= max(2, K):
                cfgs += [("PREC", 0, "matched", prec[:k]), ("NS", 0, "matched", ns[:k]),
                         ("GEO", 0, "matched", geo[:k])]
                cfgs += [("RAND", d, "matched", rm.rng_for(*keys, k, d).choice(C, k, replace=False))
                         for d in range(5)]
            if 1 <= r_star < C:
                cfgs += [("PREC", 0, "rstar", prec[:r_star]), ("NS", 0, "rstar", ns[:r_star])]
            run_cfgs(cfgs, decs, Str, ytr, Ste, yte, T, base, rows, sel_rows, full)
            fold_rows.append(dict(base, r_star=r_star, n_train=len(tr), **{
                f"n_{kk}": int(b.sum()) for kk, b in bad.items()}, **{
                f"ch_{kk}": " ".join(map(str, np.flatnonzero(b))) for kk, b in bad.items()}))


def job_ksel(dataset, subj, seed, rows, sel_rows, fold_rows):
    names = channel_names(dataset)
    geo = sel.geo_ranking(names)
    units = within_units(dataset, subj, max_sessions=3 if dataset == "stieger2021" else None)
    for sess, S, y, T in units:
        C, K = S.shape[1], len(np.unique(y))
        grid = sorted({k for k in GRID if max(K, 2) <= k < C} | {C})
        for fi, (tr, te) in enumerate(rm.folds(y, seed)):
            c = np.trace(S[tr], axis1=1, axis2=2).mean() / C
            Str, Ste, ytr, yte = S[tr] / c, S[te] / c, y[tr], y[te]
            base = dict(exp="KSEL", dataset=dataset, subject=subj, seed=seed, fold=fi, frac=1.0, sub_draw=0,
                        montage=f"session:{sess}")
            acc = {(rk, dec): np.zeros(len(grid)) for rk in ("PREC", "GEO") for dec in ("MDM", "TSLR")}
            inner = StratifiedKFold(n_splits=max(2, min(5, int(np.bincount(ytr).min()))), shuffle=True,
                                    random_state=seed + 1000)
            for itr, iva in inner.split(np.zeros(len(ytr)), ytr):
                A, B, ya, yb = Str[itr], Str[iva], ytr[itr], ytr[iva]
                orders = {"PREC": prec_rank(A), "GEO": geo}
                cache = {}
                for rk, order in orders.items():
                    for gi, k in enumerate(grid):
                        ch = np.sort(order[:k]) if k < C else np.arange(C)
                        for dec, fn in (("MDM", lambda a, b_, ch=ch: fit_predict("MDM", a, ya, b_, T)),
                                        ("TSLR", lambda a, b_, ch=ch: tslr_fixed(a, ya, b_, T))):
                            key = (dec, k) if k == C else None
                            if key is not None and key in cache:
                                acc[(rk, dec)][gi] += cache[key]
                                continue
                            try:
                                a_ = np.mean(fn(subset(A, ch), subset(B, ch)) == yb)
                            except Exception:
                                a_ = 0.
                            acc[(rk, dec)][gi] += a_
                            if key is not None:
                                cache[key] = a_
            prec = prec_rank(Str)
            sr = sel.ScoreRanker(Str, ytr)
            r_star = sr.rule["r_star"]
            full = eval_full(("MDM", "TSLR"), Str, ytr, Ste, yte, T, base, rows)
            for rk, order in (("PREC", prec), ("GEO", geo)):
                for dec in ("MDM", "TSLR"):
                    kb = grid[int(np.argmax(acc[(rk, dec)]))]
                    ch = order[:kb] if kb < C else np.arange(C)
                    run_cfgs([(rk, 0, "innercv", ch)], (dec,), Str, ytr, Ste, yte, T, base, rows, None, full)
                    fold_rows.append(dict(base, ranking=rk, decoder=dec, k_inner=kb, n_train=len(tr), C=C,
                                          inner_curve=" ".join(f"{v:.4f}" for v in acc[(rk, dec)] / inner.n_splits),
                                          grid=" ".join(map(str, grid))))
                cf = [(rk, 0, f"frac{int(f * 100)}", order[:max(K, int(round(f * C)))]) for f in (0.25, 0.5, 0.75)]
                run_cfgs(cf, ("MDM", "TSLR"), Str, ytr, Ste, yte, T, base, rows, None, full)
            ref = [("PREC", 0, "rstar", prec[:r_star])] if 1 <= r_star < C else []
            ref += [("PREC", 0, "fixed16", prec[:16])] if 16 < C else []
            run_cfgs(ref, ("MDM", "TSLR"), Str, ytr, Ste, yte, T, base, rows, None, full)


def xsess_pairs(dataset, d):
    sess = [s for s in sorted(set(d["session"].tolist()), key=sesskey)
            if len(np.unique(d["y"][d["session"] == s])) > 1
            and np.bincount(d["y"][d["session"] == s]).min() >= rm.MIN_PER_CLASS]
    if dataset == "stieger2021":
        return list(zip(sess[:-1], sess[1:]))
    if dataset == "hgd_runs":
        return [("0train", "1test")]
    return [p for p in PAIRS[dataset] if p[0] in sess and p[1] in sess]


def job_xsess(dataset, subj, seed, rows, sel_rows, fold_rows):
    names = channel_names(dataset)
    pos, ok = sphere_positions(names)
    W, wok = ransac_weights(pos, ok, rm.rng_for(subj, 77))
    geo = sel.geo_ranking(names)
    m22 = m22_idx(names)
    d = load_subject(dataset, subj, session=None)
    S_all, y_all, T = d["S"], d["y"], d["T"]
    lab = d["session"]
    if dataset == "hgd_runs":
        lab = np.load(cov_path(dataset, subj), allow_pickle=True)["run"].astype(str)
        d = dict(d, session=lab)
    decs = decoders_for(dataset, "XSESS")
    for pi, (a, b) in enumerate(xsess_pairs(dataset, d)):
        tr, te = np.flatnonzero(lab == a), np.flatnonzero(lab == b)
        C = S_all.shape[1]
        c = np.trace(S_all[tr], axis1=1, axis2=2).mean() / C
        Str, Ste, ytr, yte = S_all[tr] / c, S_all[te] / c, y_all[tr], y_all[te]
        keys = (seed, subj, pi, 0)
        base = dict(exp="HSPLIT" if dataset == "hgd_runs" else "XSESS", dataset=dataset, subject=subj, seed=seed,
                    fold=0, frac=1.0, sub_draw=0, montage=f"{a}->{b}")
        sr = sel.ScoreRanker(Str, ytr)
        sr_sh = sel.ScoreRanker(Str, rm.rng_for(*keys, 7).permutation(ytr))
        r = sr.rule["r_star"]
        prec, ns = prec_rank(Str), ns_rank(Str)
        bad = detect(Str, W, wok)["PREP_ALL"]
        cfgs = [("PREP_ALL", 0, "detect", np.flatnonzero(~bad))]
        if 1 <= r < C:
            cfgs += [("T1", 0, "rstar", sr.channels("standard", r)), ("SH", 0, "rstar", sr_sh.channels("standard", r)),
                     ("PREC", 0, "rstar", prec[:r]), ("NS", 0, "rstar", ns[:r]), ("GEO", 0, "rstar", geo[:r])]
            cfgs += [("RAND", dd, "rstar", rm.rng_for(*keys, r, dd).choice(C, r, replace=False)) for dd in range(10)]
        if 16 < C:
            cfgs += [("PREC", 0, "fixed16", prec[:16]), ("GEO", 0, "fixed16", geo[:16])]
        if 4 <= len(m22) < C:
            cfgs += [("M22", 0, "montage", m22)]
        # selection is fitted on the raw training session; decoders are evaluated without
        # adaptation and after unsupervised per-session re-centring of each channel subset
        for tf, tag in ((None, ""), (recenter, ":rc")):
            b2 = dict(base, montage=base["montage"] + tag)
            full = eval_full(decs, Str, ytr, Ste, yte, T, b2, rows, tf)
            run_cfgs(cfgs, decs, Str, ytr, Ste, yte, T, b2, rows, sel_rows if tf is None else None, full, tf)
        fold_rows.append(dict(base, r_star=r, n_train=len(tr), n_test=len(te), n_bad=int(bad.sum())))


def job_gsize(dataset, subj, seed, rows, sel_rows, fold_rows):
    d = load_subject(dataset, subj)
    S, y, T = d["S"], d["y"], d["T"]
    names = channel_names(dataset)
    idx = m22_idx(names)
    S0 = subset(S, idx)
    C0, n = len(idx), len(y)
    for fi, (tr, te) in enumerate(rm.folds(y, seed)):
        scale = float(np.median(np.diagonal(S0[tr].mean(0))))
        for ntype, sig in (("stationary", 0.0), ("nonstat10", 1.0)):
            for m in (0, 16, 32):
                if m == 0 and ntype != "stationary":
                    continue
                rs = rm.rng_for(seed, subj, m, int(sig * 10), 99)
                v = (scale * rs.chisquare(T, size=(n, m)) / T if sig == 0 else
                     scale * np.exp(sig * rs.standard_normal((n, m)) - sig ** 2 / 2))
                Sa = np.zeros((n, C0 + m, C0 + m))
                Sa[:, :C0, :C0] = S0
                Sa[:, np.arange(C0, C0 + m), np.arange(C0, C0 + m)] = v
                for frac in (0.25, 0.5, 1.0):
                    for sd in (range(3) if frac < 1 else [0]):
                        if frac < 1:
                            rsub = rm.rng_for(seed, subj, fi, int(frac * 100), sd)
                            sub = np.concatenate([rsub.choice(tr[y[tr] == cc], max(2, int(round(frac * np.sum(y[tr] == cc)))),
                                                              replace=False) for cc in np.unique(y)])
                        else:
                            sub = tr
                        c = np.trace(Sa[sub], axis1=1, axis2=2).mean() / (C0 + m)
                        Str, Ste = Sa[sub] / c, Sa[te] / c
                        base = dict(exp="GSIZE", dataset=dataset, subject=subj, seed=seed, fold=fi, frac=frac,
                                    sub_draw=sd, montage=f"{ntype}:{m}")
                        for dec in (("MDM", "TSLR", "TSLR_RAW") if frac == 1.0 else ("MDM",)):
                            nc, st, _ = evaluate(dec, Str, y[sub], Ste, y[te], T)
                            rows.append(dict(base, selector="FULL", draw=0, k_rule="full", k=C0 + m, decoder=dec,
                                             n_test=len(te), n_correct=nc, status=st))
                        if m > 0:
                            keep = prec_rank(Str)[:C0]
                            nc, st, _ = evaluate("MDM", subset(Str, keep), y[sub], subset(Ste, keep), y[te], T)
                            rows.append(dict(base, selector="PREC", draw=0, k_rule="fixed", k=C0, decoder="MDM",
                                             n_test=len(te), n_correct=nc, status=st))
                            fold_rows.append(dict(base, frac_kept_real=float(np.mean(keep < C0))))


JOBS = dict(BADCH=job_badch, KSEL=job_ksel, XSESS=job_xsess, HSPLIT=job_xsess, GSIZE=job_gsize)


def job(args):
    exp, dataset, subj, seed = args
    out_dir = RUNS / exp / dataset
    fout = out_dir / f"subj_{subj}_seed_{seed}.csv.gz"
    if fout.exists():
        return subj, seed, "skip", 0.
    t0 = time.time()
    rows, sel_rows, fold_rows = [], [], []
    JOBS[exp](dataset, subj, seed, rows, sel_rows, fold_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("sel_", sel_rows), ("folds_", fold_rows)):
        pd.DataFrame(data).to_csv(out_dir / f"{name}subj_{subj}_seed_{seed}.csv.gz", index=False, compression="gzip")
    tmp = out_dir / f".subj_{subj}_seed_{seed}.{os.getpid()}.csv.gz"
    pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
    for f in list(out_dir.glob(f"*subj_{subj}_seed_{seed}.csv.gz")) + [tmp]:
        with open(f, "rb") as fh:
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
    ap.add_argument("exp", choices=sorted(JOBS))
    ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    seeds = ([int(s) for s in a.seeds.split(",")] if a.seeds else
             [42] if a.exp in ("XSESS", "HSPLIT") else [42, 43, 44])
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
