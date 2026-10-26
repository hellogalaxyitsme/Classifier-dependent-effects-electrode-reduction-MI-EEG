"""Monte-Carlo check of Propositions 1-3 of the article.

Two classes. Informative block: p = 4 channels, trial covariances
Wishart(T, Sigma_c)/T with Sigma_1 = diag(1+a, 1-a, 1, 1), Sigma_2 = diag(1-a, 1+a, 1, 1).
Extra block: q independent channels with log-variance mu + sigma*eps, class-free.
For each (sigma, q, n) the MDM accuracy is compared with the Gaussian
prediction Phi(m / sqrt(s^2 + V)) of Proposition 2, where m and s are the
class-conditional mean and SD of the informative decision statistic
estimated in the same replicate. TS-LR accuracy is reported alongside.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mech.data import PROJ  # noqa: E402
from mech.decoders import fit_predict  # noqa: E402

from pyriemann.classification import MDM  # noqa: E402
from pyriemann.utils.distance import distance_riemann  # noqa: E402

P, T, A = 4, 200, 0.25
N_TEST = 400


def sample(n, cls, rs):
    sig = np.ones(P)
    sig[0], sig[1] = (1 + A, 1 - A) if cls == 0 else (1 - A, 1 + A)
    X = rs.standard_normal((n, P, T)) * np.sqrt(sig)[None, :, None]
    return np.einsum("nct,ndt->ncd", X, X) / T


def block(Aa, q, sigma, rs):
    n = len(Aa)
    S = np.zeros((n, P + q, P + q))
    S[:, :P, :P] = Aa
    if q:
        S[:, np.arange(P, P + q), np.arange(P, P + q)] = np.exp(sigma * rs.standard_normal((n, q)))
    return S


rows = []
rs0 = np.random.default_rng(0)
for rep in range(20):
    for n in (25, 50, 100, 200):
        rs = np.random.default_rng([rep, n])
        Atr = [sample(n, c, rs) for c in (0, 1)]
        Ate = [sample(N_TEST, c, rs) for c in (0, 1)]
        ytr = np.r_[np.zeros(n), np.ones(n)].astype(int)
        yte = np.r_[np.zeros(N_TEST), np.ones(N_TEST)].astype(int)
        mdm = MDM(metric="riemann").fit(np.concatenate(Atr), ytr)
        M1, M2 = mdm.covmeans_
        dA = [np.array([distance_riemann(x, M1) ** 2 - distance_riemann(x, M2) ** 2 for x in Ate[c]]) for c in (0, 1)]
        m = (dA[1].mean() - dA[0].mean()) / 2
        s = np.sqrt((dA[0].var() + dA[1].var()) / 2)
        for sigma in (0.0, 0.25, 0.5, 1.0):
            for q in (0, 4, 16, 32):
                if q == 0 and sigma > 0:
                    continue
                rq = np.random.default_rng([rep, n, q, int(sigma * 100)])
                Str = np.concatenate([block(Atr[c], q, sigma, rq) for c in (0, 1)])
                Ste = np.concatenate([block(Ate[c], q, sigma, rq) for c in (0, 1)])
                if sigma == 0 and q:
                    Str[:, P:, P:] = np.eye(q)
                    Ste[:, P:, P:] = np.eye(q)
                acc_mdm = np.mean(fit_predict("MDM", Str, ytr, Ste, T) == yte)
                acc_ts = np.mean(fit_predict("TSLR", Str, ytr, Ste, T) == yte)
                acc_tr = np.mean(fit_predict("TSLR_RAW", Str, ytr, Ste, T) == yte)
                pred = stats.norm.cdf(m / np.sqrt(s ** 2 + 4 * q * sigma ** 4 * (2 / n)))
                # including the variance of the training-dependent offset, 2 q sigma^4 (1/n1^2 + 1/n2^2)
                pred_full = stats.norm.cdf(m / np.sqrt(s ** 2 + 4 * q * sigma ** 4 * (2 / n)
                                                       + 2 * q * sigma ** 4 * (2 / n ** 2)))
                rows.append(dict(rep=rep, n=n, sigma=sigma, q=q, mdm=acc_mdm, tslr=acc_ts, tslr_raw=acc_tr,
                                 mdm_pred=pred, mdm_pred_full=pred_full, mdm_pred_q0=stats.norm.cdf(m / s)))
d = pd.DataFrame(rows)
out = d.groupby(["sigma", "q", "n"])[["mdm", "mdm_pred", "mdm_pred_full", "tslr", "tslr_raw",
                                      "mdm_pred_q0"]].mean().round(3)
pd.set_option("display.width", 200)
print(out.to_string())
for col in ("mdm_pred", "mdm_pred_full"):
    print(f"\n{col}: correlation with simulated MDM accuracy", round(np.corrcoef(d[col], d.mdm)[0, 1], 3),
          " mean abs error:", round(np.mean(np.abs(d[col] - d.mdm)), 4),
          " condition-mean r:", round(np.corrcoef(out[col], out.mdm)[0, 1], 4),
          " condition-mean MAE:", round(np.mean(np.abs(out[col] - out.mdm)), 4),
          " max abs difference of the two predictions:", round(np.max(np.abs(d.mdm_pred_full - d.mdm_pred)), 4))
d.to_csv(PROJ / "results" / "simulation_propositions.csv", index=False)
