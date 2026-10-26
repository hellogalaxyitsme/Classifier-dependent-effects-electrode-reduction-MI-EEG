"""Exploratory follow-up analyses (after the confirmatory analysis).

usage: python analyze_gaps.py <part> [...]

parts
  dist     per-participant distributions of the differences from the full
           montage, margin sensitivity, MDM gain as a share of the
           MDM-to-TS-LR gap, random-effects meta-analysis (confirmatory data)
  stab     selection stability within sessions (folds x split seeds) and
           across sessions
  mech     channel-level and subject-level variance-fluctuation analysis
  exp      summaries of BADCH, KSEL, XSESS, HSPLIT and GSIZE

Results are written to results/gaps_<part>*.csv.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.analysis import load, subject_level  # noqa: E402
from mech.data import PROJ, channel_names, cov_path, load_subject, subjects  # noqa: E402

RES = PROJ / "results"
CONFSETS = ["dreyer2023", "physionetmi", "yang2025", "stieger2021"]
DEVSETS = ["cho2017", "lee2019_mi", "schirrmeister2017", "bnci2014_001"]
BNCI22 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
          "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
pd.set_option("display.width", 250)


def ci90(x):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    h = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), x.mean() - h, x.mean() + h


def conf_table(ds):
    """Subject x (decoder, selector) accuracy (%), confirmatory protocol; CSP-LDA
    from the rescaled re-evaluation."""
    d = load("CONF", ds)
    d = d[d.decoder != "CSPLDA"]
    c = load("CONF_CSPNORM", ds)
    if not c.empty:
        c = c.assign(exp="CONF", status="ok", frac=1.0, sub_draw=0)
        d = pd.concat([d, c[d.columns.intersection(c.columns)]], ignore_index=True)
    s = subject_level(d)
    s = s[s.k_rule.isin(["rstar", "full", "montage"])]
    s = s.groupby(["subject", "decoder", "selector"])["acc"].mean().reset_index()
    return s.pivot_table(index="subject", columns=["decoder", "selector"], values="acc") * 100


# ---------------------------------------------------------------------------
def part_dist():
    rows, meta_in, share = [], [], []
    for ds in CONFSETS:
        w = conf_table(ds)
        for dec, sels in (("MDM", ["PREC", "GEO", "M22", "BB", "CSPF", "RAND"]), ("TSLR", ["PREC", "GEO", "M22", "BB"]),
                          ("FGMDM", ["PREC", "GEO", "M22", "BB"]), ("CSPLDA", ["PREC", "GEO", "M22", "BB"])):
            for s in sels:
                if (dec, s) not in w:
                    continue
                x = (w[(dec, s)] - w[(dec, "FULL")]).dropna()
                m, lo, hi = ci90(x)
                rows.append(dict(dataset=ds, decoder=dec, selector=s, n=len(x), mean=m, lo90=lo, hi90=hi,
                                 median=x.median(), q10=x.quantile(.1), q90=x.quantile(.9),
                                 pct_below_m5=100 * (x < -5).mean(), pct_below_m3=100 * (x < -3).mean(),
                                 pct_below_0=100 * (x < 0).mean(), pct_above_0=100 * (x > 0).mean(),
                                 pct_above_3=100 * (x > 3).mean(), pct_above_5=100 * (x > 5).mean(),
                                 **{f"noninf_{mg}": bool(lo > -mg) for mg in (0.5, 1, 2, 3)}))
                meta_in.append(dict(dataset=ds, decoder=dec, selector=s, mean=x.mean(),
                                    se=x.std(ddof=1) / np.sqrt(len(x)), n=len(x)))
        for a, b in (("PREC", "BB"), ("PREC", "CSPF"), ("M22", "PREC"), ("GEO", "PREC")):
            x = (w[("MDM", a)] - w[("MDM", b)]).dropna()
            m, lo, hi = ci90(x)
            rows.append(dict(dataset=ds, decoder="MDM", selector=f"{a}-{b}", n=len(x), mean=m, lo90=lo, hi90=hi,
                             median=x.median(), **{f"noninf_{mg}": bool(lo > -mg) for mg in (0.5, 1, 2, 3)}))
        gap = w[("TSLR", "FULL")].mean() - w[("MDM", "FULL")].mean()
        for s in ("PREC", "GEO", "M22"):
            if ("MDM", s) in w:
                share.append(dict(dataset=ds, selector=s, mdm_full=w[("MDM", "FULL")].mean(),
                                  tslr_full=w[("TSLR", "FULL")].mean(), gap=gap,
                                  gain=(w[("MDM", s)] - w[("MDM", "FULL")]).mean(),
                                  share_of_gap=(w[("MDM", s)] - w[("MDM", "FULL")]).mean() / gap,
                                  best_reduced_mdm=w[("MDM", s)].mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "followup_dist.csv", index=False)
    cols = ["dataset", "decoder", "selector", "n", "mean", "lo90", "hi90", "median", "q10", "q90",
            "pct_below_m5", "pct_below_m3", "pct_above_0", "pct_above_3", "noninf_0.5", "noninf_1", "noninf_2", "noninf_3"]
    print(out[cols].round(2).to_string(index=False))
    sh = pd.DataFrame(share)
    sh.to_csv(RES / "followup_gapshare.csv", index=False)
    print("\nMDM gain as a share of the full-montage MDM-to-TS-LR gap\n", sh.round(3).to_string(index=False))
    # DerSimonian-Laird random-effects meta-analysis over the confirmatory datasets
    mi = pd.DataFrame(meta_in)
    res = []
    for (dec, s), g in mi.groupby(["decoder", "selector"]):
        if len(g) < 2:
            continue
        y, v = g["mean"].values, g["se"].values ** 2
        wf = 1 / v
        mf = np.sum(wf * y) / wf.sum()
        Q = np.sum(wf * (y - mf) ** 2)
        dfree = len(y) - 1
        tau2 = max(0., (Q - dfree) / (wf.sum() - np.sum(wf ** 2) / wf.sum()))
        wr = 1 / (v + tau2)
        mr = np.sum(wr * y) / wr.sum()
        se = np.sqrt(1 / wr.sum())
        I2 = max(0., (Q - dfree) / Q) * 100 if Q > 0 else 0.
        res.append(dict(decoder=dec, selector=s, k=len(y), pooled=mr, lo95=mr - 1.96 * se, hi95=mr + 1.96 * se,
                        tau=np.sqrt(tau2), I2=I2, Q_p=1 - stats.chi2.cdf(Q, dfree),
                        pi_lo=mr - stats.t.ppf(.975, max(dfree - 1, 1)) * np.sqrt(tau2 + se ** 2),
                        pi_hi=mr + stats.t.ppf(.975, max(dfree - 1, 1)) * np.sqrt(tau2 + se ** 2)))
    me = pd.DataFrame(res)
    me.to_csv(RES / "followup_meta.csv", index=False)
    print("\nRandom-effects meta-analysis over the confirmatory datasets (pp)\n", me.round(2).to_string(index=False))


# ---------------------------------------------------------------------------
def nogueira(Z):
    M, d = Z.shape
    kbar = Z.sum(1).mean()
    den = (kbar / d) * (1 - kbar / d)
    return 1 - Z.var(0, ddof=1).mean() / den if den > 0 and M > 1 else np.nan


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def part_stab():
    rows = []
    for exp, sets in (("CONF", CONFSETS), ("A3", DEVSETS)):
        for ds in sets:
            files = glob.glob(str(PROJ / "runs" / exp / ds / "sel_subj_*_seed_*.csv.gz"))
            if not files:
                continue
            C = len(channel_names(ds))
            s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            s = s[s.k_rule.isin(["rstar", "fixed"]) & ((s.k_rule == "rstar") | (s.k == 16))]
            if "montage" not in s:
                s["montage"] = "first"
            for (sl, kr), g in s.groupby(["selector", "k_rule"]):
                if sl == "RAND":
                    continue
                vals, kk = [], []
                for _, gs in g.groupby(["subject", "montage"]):
                    Z = np.zeros((len(gs), C))
                    for i, ch in enumerate(gs["channels"].astype(str)):
                        Z[i, [int(v) for v in ch.split()]] = 1
                    vals.append(nogueira(Z))
                    kk.append(gs.k.mean())
                rows.append(dict(scope="within-session (folds x seeds)", dataset=ds, selector=sl, k_rule=kr,
                                 n_units=len(vals), phi=np.nanmean(vals), phi_sd=np.nanstd(vals), mean_k=np.mean(kk), C=C))
    for exp in ("XSESS",):
        for ds in ["bnci2014_001", "lee2019_2s", "yang2025", "stieger2021"]:
            files = glob.glob(str(PROJ / "runs" / exp / ds / "sel_subj_*_seed_*.csv.gz"))
            if not files:
                continue
            C = len(channel_names(ds))
            s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            s["train"] = s.montage.str.split("->").str[0]
            s = s.drop_duplicates(["subject", "train", "selector", "k_rule"])
            for (sl, kr), g in s.groupby(["selector", "k_rule"]):
                vals, js, jr = [], [], []
                for _, gs in g.groupby("subject"):
                    if len(gs) < 2:
                        continue
                    sets_ = [[int(v) for v in ch.split()] for ch in gs["channels"].astype(str)]
                    Z = np.zeros((len(sets_), C))
                    for i, ch in enumerate(sets_):
                        Z[i, ch] = 1
                    vals.append(nogueira(Z))
                    pj = [jaccard(sets_[i], sets_[j]) for i in range(len(sets_)) for j in range(i + 1, len(sets_))]
                    js.append(np.mean(pj))
                    k = np.mean([len(v) for v in sets_])
                    jr.append(k / (2 * C - k))
                if vals:
                    rows.append(dict(scope="across sessions", dataset=ds, selector=sl, k_rule=kr, n_units=len(vals),
                                     phi=np.nanmean(vals), phi_sd=np.nanstd(vals), jaccard=np.mean(js),
                                     jaccard_random=np.mean(jr), C=C))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "followup_stability.csv", index=False)
    print(out.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
def fluct(S):
    """Per-channel SD over trials of log unique (conditional) variance and of
    log variance; label-free."""
    n, C, _ = S.shape
    lam = 1e-3 * np.trace(S, axis1=1, axis2=2) / C
    P = np.linalg.inv(S + lam[:, None, None] * np.eye(C))
    lu = -np.log(np.diagonal(P, axis1=1, axis2=2))
    lv = np.log(np.diagonal(S, axis1=1, axis2=2))
    return lu.std(0), lv.std(0)


def perf_table(ds):
    if ds in CONFSETS:
        w = conf_table(ds)
        return pd.DataFrame({"mdm_prec": w[("MDM", "PREC")] - w[("MDM", "FULL")],
                             "mdm_m22": (w[("MDM", "M22")] - w[("MDM", "FULL")]) if ("MDM", "M22") in w else np.nan,
                             "tslr_prec": w[("TSLR", "PREC")] - w[("TSLR", "FULL")],
                             "gap_full": w[("TSLR", "FULL")] - w[("MDM", "FULL")]})
    a = subject_level(pd.concat([load("A", ds), load("A3", ds)], ignore_index=True).assign(exp="A"))
    a = a[a.k_rule.isin(["rstar", "full"])]
    w = a.pivot_table(index="subject", columns=["decoder", "selector"], values="acc") * 100
    return pd.DataFrame({"mdm_prec": w[("MDM", "PREC")] - w[("MDM", "FULL")], "mdm_m22": np.nan,
                         "tslr_prec": w[("TSLR", "PREC")] - w[("TSLR", "FULL")],
                         "gap_full": w[("TSLR", "FULL")] - w[("MDM", "FULL")]})


def part_mech():
    ch_rows, subj_rows = [], []
    for ds in CONFSETS + DEVSETS:
        names = channel_names(ds)
        low = [n.lower() for n in names]
        cen = np.zeros(len(names), bool)
        cen[[low.index(n.lower()) for n in BNCI22 if n.lower() in low]] = True
        exp = "CONF" if ds in CONFSETS else "A3"
        sel = pd.concat([pd.read_csv(f) for f in glob.glob(str(PROJ / "runs" / exp / ds / "sel_subj_*.csv.gz"))],
                        ignore_index=True)
        sel = sel[(sel.selector == "PREC") & (sel.k_rule == "rstar")].copy()
        if "montage" not in sel or ds in DEVSETS:
            sel["montage"] = "session:first"          # development runs: first session only
        for subj in subjects(ds):
            if ds in CONFSETS:
                d = load_subject(ds, subj, session=None)
            else:
                d = load_subject(ds, subj)
                d["session"] = np.array(["first"] * len(d["y"]))
            fu_s, fv_s = [], []
            for sess in sorted(set(d["session"].tolist())):
                gs = sel[(sel.subject == subj) & (sel.montage == f"session:{sess}")]
                if gs.empty:
                    continue
                m = d["session"] == sess
                fu, fv = fluct(d["S"][m])
                freq = np.zeros(len(names))
                for ch in gs["channels"].astype(str):
                    freq[[int(v) for v in ch.split()]] += 1
                freq /= len(gs)
                rho_u = stats.spearmanr(freq, fu).correlation
                rho_v = stats.spearmanr(freq, fv).correlation
                kept = freq >= 0.5
                ch_rows.append(dict(dataset=ds, subject=subj, session=sess, rho_keep_vs_unique_fluct=rho_u,
                                    rho_keep_vs_var_fluct=rho_v,
                                    fluct_dropped_minus_kept=fu[~kept].mean() - fu[kept].mean() if kept.any() and (~kept).any() else np.nan,
                                    fluct_periph_minus_central=fu[~cen].mean() - fu[cen].mean() if (~cen).any() else np.nan))
                fu_s.append(fu)
                fv_s.append(fv)
            if fu_s:
                fu = np.mean(fu_s, 0)
                subj_rows.append(dict(dataset=ds, subject=subj, F_all=fu.mean(),
                                      F_periph_excess=fu[~cen].mean() - fu[cen].mean() if (~cen).any() else np.nan))
    ch = pd.DataFrame(ch_rows)
    ch.to_csv(RES / "followup_fluctuation_channels.csv", index=False)
    print("Channel level (per session): Spearman rho between PREC keep-frequency and fluctuation; "
          "fluctuation of dropped minus kept channels")
    agg = ch.groupby("dataset").agg(n=("subject", "size"), rho_unique=("rho_keep_vs_unique_fluct", "mean"),
                                    frac_rho_neg=("rho_keep_vs_unique_fluct", lambda v: (v < 0).mean()),
                                    rho_var=("rho_keep_vs_var_fluct", "mean"),
                                    dropped_minus_kept=("fluct_dropped_minus_kept", "mean"),
                                    periph_minus_central=("fluct_periph_minus_central", "mean"))
    print(agg.round(3).to_string())
    sj = pd.DataFrame(subj_rows)
    res = []
    for ds, g in sj.groupby("dataset"):
        p = perf_table(ds)
        g = g.set_index("subject").join(p, how="inner")
        for idx in ("F_all", "F_periph_excess"):
            for tgt in ("mdm_prec", "mdm_m22", "tslr_prec", "gap_full"):
                x = g[[idx, tgt]].dropna()
                if len(x) < 6:
                    continue
                r = stats.spearmanr(x[idx], x[tgt])
                res.append(dict(dataset=ds, index=idx, target=tgt, n=len(x), rho=r.correlation, p=r.pvalue))
    rr = pd.DataFrame(res)
    pooled = []
    for (idx, tgt), g in rr.groupby(["index", "target"]):
        z = np.arctanh(g.rho.clip(-0.999, 0.999))
        wts = g.n - 3
        zm = np.sum(wts * z) / wts.sum()
        se = 1 / np.sqrt(wts.sum())
        pooled.append(dict(index=idx, target=tgt, datasets=len(g), n=int(g.n.sum()), rho_pooled=np.tanh(zm),
                           lo95=np.tanh(zm - 1.96 * se), hi95=np.tanh(zm + 1.96 * se),
                           p=2 * (1 - stats.norm.cdf(abs(zm / se)))))
    rr.to_csv(RES / "followup_fluctuation_subjects.csv", index=False)
    pd.DataFrame(pooled).to_csv(RES / "followup_fluctuation_pooled.csv", index=False)
    print("\nSubject level: Spearman rho between fluctuation index and gains\n", rr.round(3).to_string(index=False))
    print("\nPooled (Fisher z, weights n-3)\n", pd.DataFrame(pooled).round(3).to_string(index=False))


# ---------------------------------------------------------------------------
def read_all(files):
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_safe(exp, ds):
    """As mech.analysis.load, skipping the empty files of excluded recordings."""
    parts = []
    for f in sorted(glob.glob(str(PROJ / "runs" / exp / ds / "subj_*_seed_*.csv.gz"))):
        try:
            parts.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["montage"] = d["montage"].fillna("native")
    return d


def summarise(exp, sets, sels, decs, by_mode=False):
    out = []
    for ds in sets:
        d = load_safe(exp, ds)
        if d.empty:
            continue
        if ds == "stieger2021" and exp in ("BADCH", "KSEL"):
            # first three eligible sessions per subject (as in run_gaps.py)
            d["_s"] = d.montage.str.replace("session:", "", regex=False).astype(int)
            keep = d.groupby("subject")["_s"].transform(lambda v: v.isin(sorted(v.unique())[:3]))
            d = d[keep].drop(columns="_s")
        if by_mode:
            d["mode"] = np.where(d.montage.str.endswith(":rc"), "recentred", "plain")
            d["montage"] = d["mode"]
        s = subject_level(d)
        s = s.groupby(["subject", "montage", "decoder", "selector", "k_rule"])["acc"].mean().reset_index()
        for (mt, dec), g in s.groupby(["montage", "decoder"]):
            if dec not in decs:
                continue
            w = g.pivot_table(index="subject", columns=["selector", "k_rule"], values="acc") * 100
            if ("FULL", "full") not in w:
                continue
            for (sl, kr) in w.columns:
                if sl == "FULL" or (sels and sl not in sels):
                    continue
                m, lo, hi = ci90(w[(sl, kr)] - w[("FULL", "full")])
                out.append(dict(exp=exp, dataset=ds, mode=mt, decoder=dec, selector=sl, k_rule=kr,
                                n=w[(sl, kr)].notna().sum(), full=w[("FULL", "full")].mean(), diff=m, lo90=lo, hi90=hi))
    return pd.DataFrame(out)


def part_exp():
    allsets = CONFSETS + DEVSETS
    tabs = []
    b = summarise("BADCH", allsets, None, ("MDM", "TSLR", "FGMDM", "CSPLDA"))
    tabs.append(b)
    if not b.empty:
        print("BADCH: difference from full montage (pp)")
        print(b.pivot_table(index=["dataset", "decoder"], columns=["selector", "k_rule"], values="diff").round(1).to_string())
        fo = read_all([f for ds in allsets for f in glob.glob(str(PROJ / "runs" / "BADCH" / ds / "folds_subj_*.csv.gz"))])
        print("\nchannels flagged per fold (mean):\n",
              fo.groupby("dataset")[["n_PREP_DEV", "n_PREP_COR", "n_PREP_RAN", "n_PREP_ALL"]].mean().round(2).to_string())
    k = summarise("KSEL", allsets, None, ("MDM", "TSLR"))
    tabs.append(k)
    if not k.empty:
        print("\nKSEL: difference from full montage (pp)")
        print(k.pivot_table(index=["dataset", "decoder"], columns=["selector", "k_rule"], values="diff").round(1).to_string())
        fo = read_all([f for ds in allsets for f in glob.glob(str(PROJ / "runs" / "KSEL" / ds / "folds_subj_*.csv.gz"))])
        fo["k_frac"] = fo.k_inner / fo.C
        print("\ninner-CV chosen k (median, IQR as fraction of C; share choosing the full montage):")
        print(fo.groupby(["dataset", "ranking", "decoder"]).agg(
            k_med=("k_inner", "median"), frac_q25=("k_frac", lambda v: v.quantile(.25)),
            frac_q75=("k_frac", lambda v: v.quantile(.75)), share_full=("k_frac", lambda v: (v >= 1).mean())).round(2).to_string())
    x = summarise("XSESS", ["bnci2014_001", "lee2019_2s", "yang2025", "stieger2021"], None,
                  ("MDM", "TSLR", "FGMDM", "CSPLDA"), by_mode=True)
    tabs.append(x)
    if not x.empty:
        print("\nXSESS: difference from full montage (pp), by adaptation")
        print(x.pivot_table(index=["dataset", "mode", "decoder"], columns=["selector", "k_rule"], values="diff").round(1).to_string())
        print(x.drop_duplicates(["dataset", "mode", "decoder"])[["dataset", "mode", "decoder", "full"]].round(1).to_string(index=False))
    h = summarise("HSPLIT", ["hgd_runs"], None, ("MDM", "TSLR", "FGMDM", "CSPLDA"), by_mode=True)
    tabs.append(h)
    if not h.empty:
        print("\nHSPLIT (HGD original split): difference from full montage (pp)")
        print(h.pivot_table(index=["mode", "decoder"], columns=["selector", "k_rule"], values="diff").round(1).to_string())
        print(h.drop_duplicates(["mode", "decoder"])[["mode", "decoder", "full"]].round(1).to_string(index=False))
    g_rows = []
    for ds in DEVSETS:
        d = load_safe("GSIZE", ds)
        if d.empty:
            continue
        s = subject_level(d)
        w = s.pivot_table(index="subject", columns=["montage", "frac", "decoder", "selector"], values="acc") * 100
        for frac in (0.25, 0.5, 1.0):
            for dec in ("MDM", "TSLR", "TSLR_RAW"):
                clean = w.get(("stationary:0", frac, dec, "FULL"))
                if clean is None:
                    continue
                for mt in ("stationary:16", "stationary:32", "nonstat10:16", "nonstat10:32"):
                    if (mt, frac, dec, "FULL") in w:
                        m, lo, hi = ci90(w[(mt, frac, dec, "FULL")] - clean)
                        g_rows.append(dict(dataset=ds, frac=frac, decoder=dec, noise=mt, harm=m, lo90=lo, hi90=hi,
                                           clean=clean.mean()))
                    if dec == "MDM" and (mt, frac, "MDM", "PREC") in w:
                        m, lo, hi = ci90(w[(mt, frac, "MDM", "PREC")] - clean)
                        g_rows.append(dict(dataset=ds, frac=frac, decoder="MDM after PREC deletion", noise=mt,
                                           harm=m, lo90=lo, hi90=hi, clean=clean.mean()))
    g = pd.DataFrame(g_rows)
    if not g.empty:
        print("\nGSIZE: accuracy change caused by injected electrodes (pp), by training fraction")
        print(g.pivot_table(index=["dataset", "decoder", "noise"], columns="frac", values="harm").round(1).to_string())
        g.to_csv(RES / "followup_augmentation_size.csv", index=False)
    pd.concat([t for t in tabs if not t.empty], ignore_index=True).to_csv(RES / "followup_experiments.csv", index=False)


if __name__ == "__main__":
    for p in sys.argv[1:] or ["dist", "stab", "mech", "exp"]:
        print(f"\n################ {p}")
        dict(dist=part_dist, stab=part_stab, mech=part_mech, exp=part_exp)[p]()
