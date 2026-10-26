"""Classical decoders operating on per-trial covariance matrices.

MDM      minimum distance to the Riemannian class means; trial covariances
         with a 1e-6 relative ridge
MDM_OAS  as MDM with Oracle-Approximating-Shrinkage trial covariances
TSLR     OAS covariances, Riemannian tangent space at the training mean,
         standardisation and L2 logistic regression with the inverse
         regularisation strength chosen by 5-fold CV on the training data
CSPLDA   CSP (three filter pairs per class)
         with log-variance features and shrinkage LDA
"""

from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def ridge(S):
    C = S.shape[-1]
    tr = np.trace(S, axis1=-2, axis2=-1)
    return S + (1e-6 * tr / C)[..., None, None] * np.eye(C)


def oas(S, n_samples):
    """Oracle-approximating shrinkage of centred sample covariances (n, C, C),
    identical to sklearn.covariance.oas applied to each trial."""
    C = S.shape[-1]
    mu = np.trace(S, axis1=-2, axis2=-1) / C
    alpha = np.mean(S ** 2, axis=(-2, -1))
    num = alpha + mu ** 2
    den = (n_samples + 1) * (alpha - mu ** 2 / C)
    sh = np.where(den == 0, 1.0, np.minimum(num / np.where(den == 0, 1, den), 1.0))
    out = (1. - sh)[:, None, None] * S
    out[:, np.arange(C), np.arange(C)] += (sh * mu)[:, None]
    return out


def _mdm(Ctr, ytr, Cte):
    from pyriemann.classification import MDM
    return MDM(metric="riemann").fit(Ctr, ytr).predict(Cte)


def fit_predict(decoder, Str, ytr, Ste, T):
    """Fit on training covariances and predict test labels."""
    if decoder == "MDM":
        return _mdm(ridge(Str), ytr, ridge(Ste))
    if decoder == "MDM_OAS":
        return _mdm(oas(Str, T), ytr, oas(Ste, T))
    if decoder == "FGMDM":
        # MDM after Fisher geodesic filtering (shrinkage LDA in the tangent space)
        from pyriemann.classification import FgMDM
        return FgMDM(metric="riemann").fit(oas(Str, T), ytr).predict(oas(Ste, T))
    if decoder in ("TSLR", "TSLR_RAW"):
        # TSLR_RAW: identical except that the trial covariances get the MDM ridge
        # instead of OAS shrinkage (used to separate shrinkage from standardisation)
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        prep = (lambda S: oas(S, T)) if decoder == "TSLR" else ridge
        ts = TangentSpace(metric="riemann")
        Ftr = ts.fit_transform(prep(Str))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, max_iter=3000))
        clf.fit(Ftr, ytr)
        return clf.predict(ts.transform(prep(Ste)))
    if decoder == "CSPLDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        from .selectors import csp_filters
        Wf, _ = csp_filters(Str, ytr, m=3)

        def feats(S):
            v = np.einsum("fi,nij,fj->nf", Wf, S, Wf)
            return np.log(np.clip(v, 1e-10, None))
        sc = StandardScaler()
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(sc.fit_transform(feats(Str)), ytr)
        return lda.predict(sc.transform(feats(Ste)))
    raise ValueError(decoder)


def evaluate(decoder, Str, ytr, Ste, yte, T):
    t0 = time.perf_counter()
    try:
        yp = fit_predict(decoder, Str, ytr, Ste, T)
        status = "ok"
    except Exception as exc:
        yp = np.full_like(yte, np.unique(ytr)[0])
        status = f"error:{type(exc).__name__}"
    return int(np.sum(yp == yte)), status, time.perf_counter() - t0


def project(S, W):
    """Covariances of the projected signals W' x (W: C x k)."""
    return np.einsum("ia,nij,jb->nab", W, S, W)


def subset(S, ch):
    ch = np.asarray(ch)
    return S[:, ch[:, None], ch[None, :]]
