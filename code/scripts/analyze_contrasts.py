"""Supplementary analyses of existing participant-level results.

usage: python analyze_contrasts.py

1. Decoder-by-reduction interaction per participant,
   (MDM_reduced - MDM_full) - (TSLR_reduced - TSLR_full), for the fixed
   22-electrode montage (M22) and for PREC with the nested channel count.
2. Pooling across datasets by DerSimonian-Laird and by REML with the
   Hartung-Knapp-Sidik-Jonkman (HKSJ) interval, for every pooled estimate.
3. Matched covariance estimators: MDM with OAS against TS-LR with OAS
   (confirmatory protocol) for M22.
4. Agreement of the channel subsets of T1, its label-permuted version (SH)
   and PREC at the same count (Jaccard index, share of identical subsets).

Writes results/contrasts_per_dataset.csv, results/contrasts_pooled.csv,
results/subset_agreement.csv.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gaps import CONFSETS, DEVSETS, conf_table, load_safe  # noqa: E402
from mech.analysis import subject_level  # noqa: E402
from mech.data import PROJ  # noqa: E402

RES = PROJ / "results"
RUNS = PROJ / "runs"
OOSSETS = ["brandl2020", "guttmannflury2025_mi", "weibo2014", "grossewentrup2009"]
pd.set_option("display.width", 250)


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------
def pool(y, se):
    """DerSimonian-Laird (95 % normal CI) and REML with HKSJ (95 % t CI)."""
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    k = len(y)
    w = 1 / v
    mf = np.sum(w * y) / w.sum()
    Q = np.sum(w * (y - mf) ** 2)
    tau2_dl = max(0., (Q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum())) if k > 1 else 0.
    wr = 1 / (v + tau2_dl)
    m_dl = np.sum(wr * y) / wr.sum()
    se_dl = np.sqrt(1 / wr.sum())

    def nll(t2):
        ww = 1 / (v + t2)
        mu = np.sum(ww * y) / ww.sum()
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(ww.sum()) + np.sum(ww * (y - mu) ** 2))
    tau2_re = 0.
    if k > 1:
        r = optimize.minimize_scalar(nll, bounds=(0., max(1e-8, 100 * np.var(y) + 10 * v.max())), method="bounded")
        tau2_re = max(0., float(r.x))
        if nll(0.) <= nll(tau2_re):
            tau2_re = 0.
    ww = 1 / (v + tau2_re)
    m_re = np.sum(ww * y) / ww.sum()
    q = np.sum(ww * (y - m_re) ** 2) / (k - 1) if k > 1 else 1.
    se_hk = np.sqrt(max(q, 1.) / ww.sum())          # modified (ad hoc) HKSJ: variance never below the Wald variance
    tq = stats.t.ppf(0.975, k - 1) if k > 1 else np.nan
    return dict(k=k, dl=m_dl, dl_lo=m_dl - 1.96 * se_dl, dl_hi=m_dl + 1.96 * se_dl, tau_dl=np.sqrt(tau2_dl),
                I2=max(0., (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.,
                reml=m_re, hk_lo=m_re - tq * se_hk, hk_hi=m_re + tq * se_hk, tau_reml=np.sqrt(tau2_re))


def eff(x):
    x = pd.Series(x).dropna()
    n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.95, n - 1) * se
    t = x.mean() / se if se > 0 else np.nan
    return dict(n=n, mean=x.mean(), se=se, lo90=x.mean() - h, hi90=x.mean() + h,
                p_pos=1 - stats.t.cdf(t, n - 1), p_neg=stats.t.cdf(t, n - 1))


# ---------------------------------------------------------------------------
# participant-level tables
# ---------------------------------------------------------------------------
def wide_exp(exp, ds):
    d = load_safe(exp, ds)
    if d.empty:
        return None
    s = subject_level(d).groupby(["subject", "decoder", "selector", "k_rule"])["acc"].mean().reset_index()
    return s.pivot_table(index="subject", columns=["decoder", "selector", "k_rule"], values="acc") * 100


def conf_wide(ds):
    w = conf_table(ds)
    w.columns = pd.MultiIndex.from_tuples([(a, b, "montage" if b == "M22" else ("full" if b == "FULL" else "rstar"))
                                           for a, b in w.columns])
    return w


def contrasts():
    tabs = {}
    for ds in CONFSETS:
        tabs[("conf", ds, "CONF")] = conf_wide(ds)
    for ds in DEVSETS + CONFSETS:
        tabs[("dev" if ds in DEVSETS else "conf", ds, "KSEL")] = wide_exp("KSEL", ds)
    for ds in OOSSETS:
        tabs[("oos", ds, "OOS")] = wide_exp("OOS", ds)
    rows = []
    F = ("FULL", "full")
    for (stage, ds, src), w in tabs.items():
        if w is None:
            continue
        cfgs = [("M22", ("M22", "montage"))] if src in ("CONF", "OOS") else []
        cfgs += [("PREC_nested", ("PREC", "innercv"))] if src in ("KSEL", "OOS") else []
        for name, red in cfgs:
            need = [("MDM",) + red, ("MDM",) + F, ("TSLR",) + red, ("TSLR",) + F]
            if any(c not in w for c in need):
                continue
            dm = w[("MDM",) + red] - w[("MDM",) + F]
            dt = w[("TSLR",) + red] - w[("TSLR",) + F]
            for label, x in (("MDM", dm), ("TSLR", dt), ("interaction", dm - dt),
                             ("MDMred_minus_TSLRfull", w[("MDM",) + red] - w[("TSLR",) + F])):
                rows.append(dict(stage=stage, dataset=ds, config=name, contrast=label, **eff(x),
                                 pct_tslr_loss3=100 * float((dt < -3).mean()) if label == "TSLR" else np.nan))
            if src == "CONF":
                for dec in ("FGMDM", "CSPLDA", "MDM_OAS"):
                    if (dec,) + red in w:
                        x = w[(dec,) + red] - w[(dec,) + F]
                        rows.append(dict(stage=stage, dataset=ds, config=name, contrast=dec, **eff(x)))
                if ("MDM_OAS",) + red in w:
                    x = (w[("MDM_OAS",) + red] - w[("MDM_OAS",) + F]) - dt
                    rows.append(dict(stage=stage, dataset=ds, config=name, contrast="interaction_OAS", **eff(x)))
    return pd.DataFrame(rows)


def pooled_table(e):
    out = []
    for (stage, cfg, con), g in e.groupby(["stage", "config", "contrast"]):
        if len(g) < 2:
            continue
        out.append(dict(stage=stage, config=cfg, contrast=con, **pool(g["mean"], g["se"])))
    # out-of-sample hypotheses O1-O6 (rows of oos_tests.csv)
    t = pd.read_csv(RES / "oos_tests.csv")
    t = t[t.evaluable.astype(bool)]
    for hid, g in t.groupby("hyp"):
        out.append(dict(stage="oos", config=hid, contrast="prereg", **pool(g["mean"], g["se"])))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# subset agreement
# ---------------------------------------------------------------------------
def overlap():
    rows = []
    for tag, root in (("prereg", RUNS / "CONF"), ("normalised", RUNS / "rescaled" / "CONF")):
        for ds in CONFSETS:
            files = sorted(glob.glob(str(root / ds / "sel_subj_*_seed_*.csv.gz")))
            if not files:
                continue
            d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            d = d[d.selector.isin(["T1", "SH", "PREC"])]
            key = ["subject", "seed", "fold", "montage", "k_rule"]
            p = d.pivot_table(index=key, columns="selector", values="channels", aggfunc="first").dropna()
            for k_rule, g in p.groupby(level="k_rule"):
                for a, b in (("T1", "SH"), ("T1", "PREC")):
                    A = g[a].map(lambda s: set(str(s).split()))
                    B = g[b].map(lambda s: set(str(s).split()))
                    jac = np.array([len(x & y) / len(x | y) for x, y in zip(A, B)])
                    same = np.array([x == y for x, y in zip(A, B)])
                    subj_j = pd.Series(jac, index=g.index.get_level_values("subject")).groupby(level=0).mean()
                    rows.append(dict(analysis=tag, dataset=ds, k_rule=k_rule, pair=f"{a}-{b}", n_folds=len(jac),
                                     n_permutations=len(jac) if b == "SH" else np.nan,
                                     jaccard_mean=jac.mean(), jaccard_p10=np.percentile(jac, 10),
                                     jaccard_min=jac.min(), identical_pct=100 * same.mean(),
                                     subj_jaccard_min=subj_j.min()))
    return pd.DataFrame(rows)


def xnest(exp="XNEST", out_name="cross_session_nested.csv"):
    """Cross-session transfer with the count chosen in the training session.

    exp="XNESTL" summarises the sensitivity run with loaded re-centring, for
    all participants and for the participants of the main run ("common")."""
    rows = []
    for ds in ["lee2019_2s", "yang2025", "stieger2021", "bnci2014_001"]:
        d = load_safe(exp, ds)
        if d.empty:
            continue
        d["mode"] = np.where(d.montage.str.endswith(":rc"), "recentred", "plain")
        groups = [("all", d)]
        if exp != "XNEST":
            main_subj = set(load_safe("XNEST", ds).subject.unique())
            groups.append(("common", d[d.subject.isin(main_subj)]))
        for grp, dd in groups:
            for mode, g in dd.groupby("mode"):
                s = subject_level(g)                      # per participant and session pair
                s = s.groupby(["subject", "decoder", "selector", "k_rule"])["acc"].mean().reset_index()
                w = s.pivot_table(index="subject", columns=["decoder", "selector", "k_rule"], values="acc") * 100
                for dec in ("MDM", "TSLR"):
                    for sel, kr in (("PREC", "innercv"), ("M22", "montage")):
                        if (dec, sel, kr) in w:
                            rows.append(dict(group=grp, dataset=ds, mode=mode, decoder=dec, selector=sel,
                                             k_rule=kr, full=w[(dec, "FULL", "full")].mean(),
                                             **eff(w[(dec, sel, kr)] - w[(dec, "FULL", "full")])))
                if ("MDM", "PREC", "innercv") in w:
                    rows.append(dict(group=grp, dataset=ds, mode=mode, decoder="MDM",
                                     selector="PREC-minus-TSLRfull", k_rule="innercv",
                                     **eff(w[("MDM", "PREC", "innercv")] - w[("TSLR", "FULL", "full")])))
                if ("MDM", "M22", "montage") in w:
                    rows.append(dict(group=grp, dataset=ds, mode=mode, decoder="MDM",
                                     selector="M22-minus-TSLRfull", k_rule="montage",
                                     **eff(w[("MDM", "M22", "montage")] - w[("TSLR", "FULL", "full")])))
        f = pd.concat([pd.read_csv(x) for x in glob.glob(str(RUNS / exp / ds / "folds_subj_*.csv.gz"))
                       if pd.read_csv(x).shape[0]], ignore_index=True)
        rows.append(dict(dataset=ds, mode="count", decoder="MDM", selector="k_inner_median",
                         mean=f.k_mdm.median(), lo90=f.k_mdm.quantile(.25), hi90=f.k_mdm.quantile(.75)))
        rows.append(dict(dataset=ds, mode="count", decoder="TSLR", selector="k_inner_median",
                         mean=f.k_tslr.median(), lo90=f.k_tslr.quantile(.25), hi90=f.k_tslr.quantile(.75),
                         full=100 * float((f.k_tslr == f.C).mean())))
    out = pd.DataFrame(rows)
    out.to_csv(RES / out_name, index=False)
    print(out.round(2).to_string(index=False))


def gcor():
    """Correlated covariance-block augmentation: change from the clean 22-electrode montage."""
    rows = []
    for ds in DEVSETS:
        d = load_safe("GCOR", ds)
        if d.empty:
            continue
        s = subject_level(d)
        w = s.pivot_table(index="subject", columns=["montage", "decoder", "selector"], values="acc") * 100
        for mont in sorted({c[0] for c in w.columns if c[0] != "clean"}):
            for dec, sel in (("MDM", "FULL"), ("MDM", "PREC"), ("MDM", "ORIG"), ("TSLR", "FULL"), ("TSLR_RAW", "FULL")):
                if (mont, dec, sel) in w:
                    rows.append(dict(dataset=ds, condition=mont, decoder=dec, selector=sel,
                                     clean=w[("clean", dec, "FULL")].mean(),
                                     **eff(w[(mont, dec, sel)] - w[("clean", dec, "FULL")])))
        f = pd.concat([pd.read_csv(x) for x in glob.glob(str(RUNS / "GCOR" / ds / "folds_subj_*.csv.gz"))],
                      ignore_index=True)
        for mont, g in f.groupby("montage"):
            rows.append(dict(dataset=ds, condition=mont, decoder="-", selector="share_kept_original",
                             mean=g.frac_kept_orig.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "augmentation_correlated.csv", index=False)
    print(out.round(2).to_string(index=False))


def main():
    if len(sys.argv) > 1:
        {"xnest": xnest, "gcor": gcor,
         "xnestl": lambda: xnest("XNESTL", "cross_session_nested_loaded.csv")}[sys.argv[1]]()
        return
    e = contrasts()
    e.to_csv(RES / "contrasts_per_dataset.csv", index=False)
    print(e.round(2).to_string(index=False))
    pt = pooled_table(e)
    pt.to_csv(RES / "contrasts_pooled.csv", index=False)
    print()
    print(pt.round(2).to_string(index=False))
    o = overlap()
    o.to_csv(RES / "subset_agreement.csv", index=False)
    print()
    print(o.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
