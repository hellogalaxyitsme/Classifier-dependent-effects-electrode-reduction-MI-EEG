"""Band-limited stationary control, size-matched random subsets and runtime
benchmark (Supplementary Sections S6, S7 and S10; Tables S9, S12 and S15).

usage: python run_controls.py <part> [--workers N]

bandlimited    32 stationary white-Gaussian channels generated at each
               development dataset's sampling rate and epoch length, filtered
               with the 8-30 Hz Butterworth band-pass used for the recordings
               and appended to M22 as diagonal variances (bl_diag) or as the
               full 32 x 32 sample covariance (bl_full); MDM, MDM after PREC
               deletion, TS-LR with OAS and unshrunk TS-LR (TSLR_RAW)
random_m22     M22 against random subsets of the same size on identical folds
               (confirmatory datasets: all eligible sessions; additional
               datasets: first eligible session)
random_nested  PREC with the nested count against random subsets of the count
               chosen in the same outer fold (counts and accuracies of FULL and
               PREC read from runs/KSEL and runs/OOS)
timing         single-threaded training and inference time of MDM and TS-LR,
               full montage against M22, from precomputed covariances (fold 0
               of the first ten participants of Cho 2017 and HGD; one warm-up
               run, then the median of five timed repetitions)

Random subsets: 10 draws per fold for MDM and 5 for TS-LR, split seeds 42-44.
Covariances of each fold are divided by the mean training-set channel
variance, as in run_gaps.py. Outputs are written to $P/results.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")        # before numpy is imported

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gaps as rg  # noqa: E402
import run_mech as rm  # noqa: E402
from mech.data import DEV, PROJ, bandpass, channel_names, load_subject, subjects  # noqa: E402
from mech.decoders import evaluate, oas, ridge, subset  # noqa: E402

OUT = Path(os.environ.get("OUT_DIR", PROJ / "results"))
DEV_SETS = ("cho2017", "lee2019_mi", "schirrmeister2017", "bnci2014_001")
SEEDS = (42, 43, 44)
N_MDM, N_TSLR = 10, 5


def summarize_paired(vals):
    x = pd.Series(vals, dtype=float).dropna()
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    h90 = float(stats.t.ppf(0.95, n - 1) * se) if n > 1 else np.nan
    h95 = float(stats.t.ppf(0.975, n - 1) * se) if n > 1 else np.nan
    _, p2 = stats.ttest_1samp(x, 0.0) if n > 1 else (np.nan, np.nan)
    return dict(n=n, mean=m, se=se, lo90=m - h90, hi90=m + h90, lo95=m - h95, hi95=m + h95, p_two=float(p2))


def pool_run(fn, jobs, workers):
    rows = []
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for r in pool.imap_unordered(fn, jobs):
            rows.append(r)
    return rows


# ---------------------------------------------------------------- band-limited
def _work_bl(args):
    ds, subj, seed = args
    d = load_subject(ds, subj)
    S, y, T = d["S"], d["y"], int(d["T"])
    fs = float(DEV[ds]["fs"])
    idx = rg.m22_idx(channel_names(ds))
    S0 = subset(S, idx)
    C0, n, m = len(idx), len(y), 32
    pad = int(round(fs))                  # 1 s on each side, trimmed after filtering
    rows, sig_rows = [], []
    for fi, (tr, te) in enumerate(rm.folds(y, seed)):
        scale = float(np.median(np.diagonal(S0[tr].mean(0))))
        rs = rm.rng_for(seed, subj, fi, 32, 701)
        X = rs.standard_normal((n, m, T + 2 * pad))
        Xf = bandpass(X, 8.0, 30.0, fs)[:, :, pad:pad + T]
        Xf -= Xf.mean(-1, keepdims=True)
        D_full = np.einsum("nct,ndt->ncd", Xf, Xf) / T
        D_full *= scale / float(np.diagonal(D_full[tr], axis1=1, axis2=2).mean())
        diag_d = np.diagonal(D_full, axis1=1, axis2=2)
        sigma = float(np.std(np.log(np.maximum(diag_d, 1e-30)), axis=0).mean())
        sig_rows.append(dict(dataset=ds, subject=subj, seed=seed, fold=fi, T=T, fs=fs,
                             sigma_iid=float(np.sqrt(2.0 / T)), sigma_bl=sigma, T_eff=2.0 / sigma ** 2))
        D_diag = np.zeros_like(D_full)
        D_diag[:, np.arange(m), np.arange(m)] = diag_d
        conds = [("clean:0", S0)]
        for cname, blk in (("bl_diag:32", D_diag), ("bl_full:32", D_full)):
            Sa = np.zeros((n, C0 + m, C0 + m))
            Sa[:, :C0, :C0] = S0
            Sa[:, C0:, C0:] = blk
            conds.append((cname, Sa))
        for cname, Sa in conds:
            Ctot = Sa.shape[1]
            c = np.trace(Sa[tr], axis1=1, axis2=2).mean() / Ctot
            Str, Ste = Sa[tr] / c, Sa[te] / c
            for dec in ("MDM", "TSLR", "TSLR_RAW"):
                nc, _, _ = evaluate(dec, Str, y[tr], Ste, y[te], T)
                rows.append(dict(dataset=ds, subject=subj, seed=seed, fold=fi, cond=cname,
                                 selector="FULL", decoder=dec, n_test=len(te), n_correct=nc))
            if Ctot > C0:
                keep = rg.prec_rank(Str)[:C0]
                nc, _, _ = evaluate("MDM", subset(Str, keep), y[tr], subset(Ste, keep), y[te], T)
                rows.append(dict(dataset=ds, subject=subj, seed=seed, fold=fi, cond=cname,
                                 selector="PREC", decoder="MDM", n_test=len(te), n_correct=nc))
    return rows, sig_rows


def run_bandlimited(workers):
    jobs = [(ds, s, sd) for ds in DEV_SETS for sd in SEEDS for s in subjects(ds)]
    res = pool_run(_work_bl, jobs, workers)
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    sf = pd.DataFrame([r for _, sig in res for r in sig])
    df.to_csv(OUT / "bandlimited_raw.csv.gz", index=False, compression="gzip")
    sf.to_csv(OUT / "bandlimited_sigma.csv", index=False)
    sub = df.groupby(["dataset", "subject", "seed", "cond", "selector", "decoder"])[["n_correct", "n_test"]].sum().reset_index()
    sub["acc"] = 100.0 * sub["n_correct"] / sub["n_test"]
    sub = sub.groupby(["dataset", "subject", "cond", "selector", "decoder"])["acc"].mean().reset_index()
    clean = sub[sub["cond"] == "clean:0"][["dataset", "subject", "decoder", "acc"]].rename(columns={"acc": "acc_clean"})
    out = []
    for ds in DEV_SETS:
        sig = sf[sf.dataset == ds]
        for cond in ("bl_diag:32", "bl_full:32"):
            for sel, dec in (("FULL", "MDM"), ("PREC", "MDM"), ("FULL", "TSLR"), ("FULL", "TSLR_RAW")):
                cur = sub[(sub.dataset == ds) & (sub.cond == cond) & (sub.selector == sel) & (sub.decoder == dec)]
                mrg = cur.merge(clean, on=["dataset", "subject", "decoder"])
                out.append(dict(dataset=ds, cond=cond, selector=sel, decoder=dec,
                                clean_mean=float(mrg["acc_clean"].mean()), sigma_iid=float(sig["sigma_iid"].mean()),
                                sigma_bl=float(sig["sigma_bl"].mean()), T_eff=float(sig["T_eff"].mean()),
                                **summarize_paired(mrg["acc"] - mrg["acc_clean"])))
    pd.DataFrame(out).to_csv(OUT / "bandlimited_summary.csv", index=False)


# ---------------------------------------------------------- random subsets
def _row(ds, subj, seed, sess, fi, dec, cfg, draw, k, nt, nc):
    return dict(dataset=ds, subject=subj, seed=seed, session=str(sess), fold=fi, decoder=dec,
                cfg=cfg, draw=draw, k=k, n_test=nt, n_correct=nc)


def _work_rand_m22(args):
    ds, subj, seed, max_sess = args
    m22 = rg.m22_idx(channel_names(ds))
    rows = []
    for sess, S, y, T in rg.within_units(ds, subj, max_sessions=max_sess):
        C, k = S.shape[1], len(m22)
        if not (18 <= k < C):
            continue
        for fi, (tr, te) in enumerate(rm.folds(y, seed)):
            c = np.trace(S[tr], axis1=1, axis2=2).mean() / C
            Str, Ste, ytr, yte = S[tr] / c, S[te] / c, y[tr], y[te]
            keys = (seed, subj, fi, zlib.crc32(str(sess).encode()) % 100000)
            for dec, nd in (("MDM", N_MDM), ("TSLR", N_TSLR)):
                nc_f, _, _ = evaluate(dec, Str, ytr, Ste, yte, T)
                nc_m, _, _ = evaluate(dec, subset(Str, m22), ytr, subset(Ste, m22), yte, T)
                rows += [_row(ds, subj, seed, sess, fi, dec, "FULL", 0, C, len(te), nc_f),
                         _row(ds, subj, seed, sess, fi, dec, "M22", 0, k, len(te), nc_m)]
                for dd in range(nd):
                    ch = np.sort(rm.rng_for(*keys, k, dd, 801).choice(C, k, replace=False))
                    nc_r, _, _ = evaluate(dec, subset(Str, ch), ytr, subset(Ste, ch), yte, T)
                    rows.append(_row(ds, subj, seed, sess, fi, dec, "RAND_M22", dd, k, len(te), nc_r))
    return rows


def _work_rand_nested(args):
    ds, subj, seed, src, max_sess = args
    path = PROJ / "runs" / src / ds / f"subj_{subj}_seed_{seed}.csv.gz"
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        sf = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    sf = sf[sf["selector"].isin(["FULL", "PREC"]) & sf["k_rule"].isin(["full", "innercv"])
            & sf["decoder"].isin(["MDM", "TSLR"])]
    full, prec = {}, {}
    for _, r in sf.iterrows():
        key = (str(r["montage"]).replace("session:", ""), int(r["fold"]), str(r["decoder"]))
        if r["selector"] == "FULL":
            full[key] = (int(r["n_test"]), int(r["n_correct"]))
        elif r["k_rule"] == "innercv":
            prec[key] = (int(r["k"]), int(r["n_test"]), int(r["n_correct"]))
    rows = []
    for sess, S, y, T in rg.within_units(ds, subj, max_sessions=max_sess):
        C = S.shape[1]
        for fi, (tr, te) in enumerate(rm.folds(y, seed)):
            c = np.trace(S[tr], axis1=1, axis2=2).mean() / C
            Str, Ste, ytr, yte = S[tr] / c, S[te] / c, y[tr], y[te]
            keys = (seed, subj, fi, zlib.crc32(str(sess).encode()) % 100000)
            for dec, nd in (("MDM", N_MDM), ("TSLR", N_TSLR)):
                key = (str(sess), fi, dec)
                if key not in prec or key not in full:
                    continue
                nt_f, nc_f = full[key]
                k_in, nt_p, nc_p = prec[key]
                assert nt_f == len(te) and nt_p == len(te), f"test-set size differs: {ds} {subj} {key}"
                rows += [_row(ds, subj, seed, sess, fi, dec, "FULL", 0, C, len(te), nc_f),
                         _row(ds, subj, seed, sess, fi, dec, "PREC_NEST", 0, k_in, len(te), nc_p)]
                for dd in range(nd):
                    if k_in >= C:              # nested count kept every electrode
                        nc_r = nc_f
                    else:
                        ch = np.sort(rm.rng_for(*keys, k_in, dd, 802).choice(C, k_in, replace=False))
                        nc_r, _, _ = evaluate(dec, subset(Str, ch), ytr, subset(Ste, ch), yte, T)
                    rows.append(_row(ds, subj, seed, sess, fi, dec, "RAND_NEST", dd, k_in, len(te), nc_r))
    return rows


def aggregate_matched(df, target, rand):
    g = df.groupby(["dataset", "subject", "session", "seed", "decoder", "cfg", "draw"])[["n_correct", "n_test"]].sum().reset_index()
    g["acc"] = 100.0 * g["n_correct"] / g["n_test"]
    g = g.groupby(["dataset", "subject", "session", "seed", "decoder", "cfg"])["acc"].mean().reset_index()
    g = g.groupby(["dataset", "subject", "decoder", "cfg"])["acc"].mean().reset_index()
    piv = g.pivot_table(index=["dataset", "subject", "decoder"], columns="cfg", values="acc").reset_index()
    out = []
    for (ds, dec), grp in piv.groupby(["dataset", "decoder"]):
        for name, a, b in ((f"{target}-FULL", target, "FULL"), (f"{rand}-FULL", rand, "FULL"),
                           (f"{target}-{rand}", target, rand)):
            out.append(dict(dataset=ds, decoder=dec, contrast=name, **summarize_paired(grp[a] - grp[b])))
    return pd.DataFrame(out), piv


def run_random(part, workers):
    if part == "random_m22":
        specs = [("dreyer2023", None), ("physionetmi", None), ("yang2025", None), ("stieger2021", None),
                 ("brandl2020", 1), ("guttmannflury2025_mi", 1), ("weibo2014", 1)]
        jobs = [(ds, s, sd, ms) for ds, ms in specs for sd in SEEDS for s in subjects(ds)]
        fn, target, rand, stem = _work_rand_m22, "M22", "RAND_M22", "random_m22"
    else:
        specs = [("schirrmeister2017", "KSEL", None), ("grossewentrup2009", "OOS", 1), ("stieger2021", "KSEL", 3),
                 ("yang2025", "KSEL", None), ("brandl2020", "OOS", 1), ("cho2017", "KSEL", None),
                 ("lee2019_mi", "KSEL", None), ("dreyer2023", "KSEL", None), ("physionetmi", "KSEL", None),
                 ("guttmannflury2025_mi", "OOS", 1), ("weibo2014", "OOS", 1)]
        jobs = [(ds, s, sd, src, ms) for ds, src, ms in specs for sd in SEEDS for s in subjects(ds)]
        fn, target, rand, stem = _work_rand_nested, "PREC_NEST", "RAND_NEST", "random_nested"
    df = pd.DataFrame([r for rows in pool_run(fn, jobs, workers) for r in rows])
    df.to_csv(OUT / f"{stem}_raw.csv.gz", index=False, compression="gzip")
    summ, piv = aggregate_matched(df, target, rand)
    summ.to_csv(OUT / f"{stem}_summary.csv", index=False)
    piv.to_csv(OUT / f"{stem}_subjects.csv", index=False)


# ------------------------------------------------------------------- timing
def run_timing(n_rep=5, n_subj=10):
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_info, threadpool_limits

    def once(Str, ytr, Ste, T):
        t0 = time.perf_counter()
        clf_m = MDM(metric="riemann").fit(ridge(Str), ytr)
        t1 = time.perf_counter()
        clf_m.predict(ridge(Ste))
        t2 = time.perf_counter()
        ts = TangentSpace(metric="riemann")
        F = ts.fit_transform(oas(Str, T))
        clf_t = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, max_iter=3000))
        clf_t.fit(F, ytr)
        t3 = time.perf_counter()
        clf_t.predict(ts.transform(oas(Ste, T)))
        t4 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2, t4 - t3), F.nbytes

    rows = []
    with threadpool_limits(limits=1):
        print("thread pools:", [(p["internal_api"], p["num_threads"]) for p in threadpool_info()], flush=True)
        for ds in ("cho2017", "schirrmeister2017"):
            m22 = rg.m22_idx(channel_names(ds))
            for s in subjects(ds)[:n_subj]:
                d = load_subject(ds, s)
                S, y, T = d["S"], d["y"], int(d["T"])
                tr, te = next(iter(rm.folds(y, 42)))
                for mname, ch in (("FULL", np.arange(S.shape[1])), ("M22", m22)):
                    Ssub = subset(S, ch)
                    c = np.trace(Ssub[tr], axis1=1, axis2=2).mean() / len(ch)
                    Str, Ste = Ssub[tr] / c, Ssub[te] / c
                    once(Str, y[tr], Ste, T)               # warm-up
                    reps = []
                    for _ in range(n_rep):
                        t, nbytes = once(Str, y[tr], Ste, T)
                        reps.append(t)
                    med = np.median(np.array(reps), axis=0) * 1e3
                    rows.append(dict(dataset=ds, subject=s, montage=mname, C=len(ch), n_train=len(tr), n_test=len(te),
                                     d_ts=len(ch) * (len(ch) + 1) // 2, feat_kib=nbytes / 1024.0,
                                     mdm_fit_ms=med[0], mdm_pred_ms_trial=med[1] / len(te),
                                     tslr_fit_ms=med[2], tslr_pred_ms_trial=med[3] / len(te)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "timing_raw.csv", index=False)
    cols = ["feat_kib", "mdm_fit_ms", "mdm_pred_ms_trial", "tslr_fit_ms", "tslr_pred_ms_trial"]
    g = df.groupby(["dataset", "montage", "C", "d_ts"])[cols]
    summ = pd.concat([g.median().add_suffix("_median"), g.quantile(0.25).add_suffix("_q25"),
                      g.quantile(0.75).add_suffix("_q75")], axis=1).reset_index()
    summ.to_csv(OUT / "timing_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["bandlimited", "random_m22", "random_nested", "timing"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "bandlimited":
        run_bandlimited(a.workers)
    elif a.part == "timing":
        run_timing()
    else:
        run_random(a.part, a.workers)
