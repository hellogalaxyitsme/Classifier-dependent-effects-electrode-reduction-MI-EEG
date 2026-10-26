"""Numerical checks reported in Supplementary Sections S3 and S9.

usage: python check_numerics.py floors    # absolute floors after unit normalisation
       python check_numerics.py cspfloor  # CSP-LDA in the folds where its floor is reached
       python check_numerics.py spd       # sessions whose Riemannian mean cannot be computed

floors  For every training fold of the confirmatory protocol, the channel score
        (T1) and the CSP-LDA features are computed after dividing the
        covariances by (a) the participant-level scalar of the unit-normalised
        analysis and (b) the mean training-set channel variance of the fold.
        Both procedures are invariant to this scalar except through absolute
        floors (1e-12 on the whitening eigenvalues and projected variances of
        T1, 1e-10 on the CSP-LDA variances); the check records whether a floor
        is reached and whether r* and the T1 subset at r* agree between (a)
        and (b). Output: results/floor_check.csv.
spd     For the participants excluded from the cross-session analyses, reports
        per session the smallest eigenvalue of the trial covariances relative
        to their mean variance, their numerical rank, non-finite values and
        zero-variance channels, and whether the Riemannian session mean can be
        computed with and without the diagonal loading of MDM.
        Output: results/spd_check.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gaps as rg  # noqa: E402
import run_mech as rm  # noqa: E402
from mech.data import cov_path, load_subject, subjects  # noqa: E402
from mech.selectors import ScoreRanker, class_stats, csp_filters  # noqa: E402

RES = rm.PROJ / "results"
CONF = ["dreyer2023", "physionetmi", "yang2025", "stieger2021"]
EXCLUDED = {"lee2019_2s": [27, 50], "yang2025": [3, 12, 18, 30, 35, 44], "stieger2021": [25, 60]}


def floors():
    rows = []
    for ds in CONF:
        for subj in subjects(ds):
            raw = np.load(cov_path(ds, subj), allow_pickle=True)
            fs = float(raw["fs"]) if "fs" in raw.files else np.nan
            if ds == "physionetmi" and fs != 160.:
                continue
            d = load_subject(ds, subj, session=None)
            S_all, y_all = d["S"], d["y"]
            C = S_all.shape[1]
            cp = np.trace(S_all, axis1=1, axis2=2).mean() / C
            for sess in sorted(set(d["session"].tolist())):
                msk = d["session"] == sess
                S, y = S_all[msk], y_all[msk]
                if len(np.unique(y)) < 2 or np.bincount(y).min() < rm.MIN_PER_CLASS:
                    continue
                for seed in (42, 43, 44):
                    for fi, (tr, te) in enumerate(rm.folds(y, seed)):
                        cf = np.trace(S[tr], axis1=1, axis2=2).mean() / C
                        out = dict(dataset=ds, subject=subj, session=sess, seed=seed, fold=fi, cf_over_cp=cf / cp)
                        res = {}
                        for tag, c in (("p", cp), ("f", cf)):
                            Str = S[tr] / c
                            _, _, cc, mix = class_stats(Str, y[tr])
                            reg = 1e-4 * np.trace(mix) / C
                            ev, evc = np.linalg.eigh(mix + reg * np.eye(C))
                            W = evc / np.sqrt(np.maximum(ev, 1e-12))
                            V = np.einsum("ij,kil,lj->kj", W, cc, W)
                            sr = ScoreRanker(Str, y[tr])
                            r = sr.rule["r_star"]
                            res[tag] = (r, tuple(np.sort(sr.channels("standard", r))))
                            out[f"ev_floor_{tag}"] = bool((ev < 1e-12).any())
                            out[f"V_floor_{tag}"] = bool((V < 1e-12).any())
                            out[f"minV_{tag}"] = float(V.min())
                            Wf, _ = csp_filters(Str, y[tr], m=3)
                            v = np.einsum("fi,nij,fj->nf", Wf, S / c, Wf)
                            out[f"csp_floor_{tag}"] = bool((v < 1e-10).any())
                            out[f"min_csp_{tag}"] = float(v.min())
                        out["same_rstar"] = res["p"][0] == res["f"][0]
                        out["same_subset"] = res["p"] == res["f"]
                        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "floor_check.csv", index=False)
    print(df.groupby("dataset")[["ev_floor_p", "V_floor_p", "csp_floor_p", "V_floor_f", "csp_floor_f",
                                 "same_rstar", "same_subset"]].mean())


def cspfloor():
    """CSP-LDA accuracy with the participant-level and the training-fold scalar
    in the folds in which the CSP-LDA floor was reached (results/floor_check.csv)."""
    from mech.decoders import evaluate
    fc = pd.read_csv(RES / "floor_check.csv")
    fc = fc[fc.csp_floor_p | fc.csp_floor_f]
    rows = []
    for (ds, subj), g in fc.groupby(["dataset", "subject"]):
        d = load_subject(ds, subj, session=None)
        S_all, y_all, T = d["S"], d["y"], d["T"]
        C = S_all.shape[1]
        cp = np.trace(S_all, axis1=1, axis2=2).mean() / C
        for _, r in g.iterrows():
            msk = d["session"] == str(r.session)
            S, y = S_all[msk], y_all[msk]
            tr, te = list(rm.folds(y, int(r.seed)))[int(r.fold)]
            cf = np.trace(S[tr], axis1=1, axis2=2).mean() / C
            acc = {}
            for tag, c in (("p", cp), ("f", cf)):
                nc, st, _ = evaluate("CSPLDA", S[tr] / c, y[tr], S[te] / c, y[te], T)
                acc[tag] = nc
            rows.append(dict(dataset=ds, subject=subj, session=r.session, seed=r.seed, fold=r.fold,
                             n_test=len(te), correct_p=acc["p"], correct_f=acc["f"]))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "floor_check_csplda.csv", index=False)
    print(len(df), "folds; folds with different correct counts:", int((df.correct_p != df.correct_f).sum()))
    pa = df.groupby("subject")[["correct_p", "correct_f", "n_test"]].sum()
    diff = 100 * (pa.correct_p - pa.correct_f) / pa.n_test
    print("participant-level difference over these folds (pp): mean %.3f, max |.| %.3f" % (diff.mean(), diff.abs().max()))
    print("total difference in correct trials:", int((df.correct_p - df.correct_f).sum()), "of", int(df.n_test.sum()))


def spd():
    from pyriemann.utils.mean import mean_riemann
    rows = []
    for ds, subs in EXCLUDED.items():
        for s in subs:
            d = load_subject(ds, s, session=None)
            S, lab = d["S"], d["session"]
            C = S.shape[1]
            used = sorted({x for p in rg.xsess_pairs(ds, d) for x in p}, key=rg.sesskey)
            for se in used:
                X = S[lab == se]
                tr = np.trace(X, axis1=1, axis2=2) / C
                ev = np.linalg.eigvalsh(X)[:, 0] / tr
                out = dict(dataset=ds, subject=s, session=se, n_trials=len(X),
                           nonfinite=int((~np.isfinite(X)).any(axis=(1, 2)).sum()),
                           zero_variance_channels=int((np.diagonal(X, axis1=1, axis2=2) <= 0).any(0).sum()),
                           min_rel_eig=float(ev.min()), trials_nonpositive=int((ev <= 0).sum()),
                           min_rank=int(min(np.linalg.matrix_rank(x) for x in X)), C=C)
                for tag, Y in (("mean_ok", X), ("mean_ok_loaded", X + 1e-6 * tr[:, None, None] * np.eye(C))):
                    try:
                        mean_riemann(Y)
                        out[tag] = True
                    except ValueError:
                        out[tag] = False
                rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "spd_check.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    {"floors": floors, "cspfloor": cspfloor, "spd": spd}[sys.argv[1]]()
