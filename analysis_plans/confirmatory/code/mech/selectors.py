"""Channel rankings and spatial projections computed from training covariances.

All functions take per-trial covariances S (n, C, C) of the training
partition and labels y, and never see test data.

Rankings (physical channel subsets)
    T1      information-theoretic score (Jensen gap + loading term), channels
            ranked by squared whitening loadings on the top-k directions
    JG      Jensen-gap (contrast-only) score, same channel mapping
    LO      loading term only (label-free: depends on the mixture spectrum)
    SH      T1 score computed with permuted training labels
    CSPF    summed squared CSP filter weights (baseline)
    CSPP    summed squared, unit-normalised CSP patterns (Haufe et al. 2014)
    BB      backward elimination maximising the summed Riemannian distance
            between class means (Barachant and Bonnet 2011)
    GEO     distance to the nearest of C3, Cz, C4 (label- and data-free)
    RAND    uniformly random subsets

Projections (all electrodes retained, k spatial components)
    PT1     top-k whitening directions by the T1 score
    PJG     top-k whitening directions by the Jensen-gap score
    PPCA    top-k principal components of the training mixture covariance
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Information-theoretic spatial score (T1)
# ---------------------------------------------------------------------------
def class_stats(S, y):
    classes = np.unique(y)
    priors = np.array([np.mean(y == c) for c in classes])
    cc = np.stack([S[y == c].mean(0) for c in classes])
    mix = np.einsum("k,kij->ij", priors, cc)
    return classes, priors, cc, mix


def direction_scores(S, y):
    """Per-direction scores in the loaded whitening basis.

    Returns dict with W (C x C whitening columns), the standard score m, its
    Jensen part, its loading part (standard = jensen + loading), and the
    eigenvalues of the loaded mixture covariance.
    """
    classes, priors, cc, mix = class_stats(S, y)
    C = S.shape[1]
    reg = 1e-4 * np.trace(mix) / C
    ev, evc = np.linalg.eigh(mix + reg * np.eye(C))
    W = evc * (1. / np.sqrt(np.maximum(ev, 1e-12)))
    V = np.maximum(np.einsum("ij,kil,lj->kj", W, cc, W), 1e-12)        # (K, C)
    standard = -0.5 * (priors[:, None] * np.log2(V)).sum(0)
    loading = -0.5 * np.log2((priors[:, None] * V).sum(0))
    jensen = standard - loading
    return dict(W=W, standard=standard, jensen=jensen, loading=loading, ev=ev, reg=reg,
                K=len(classes))


def count_rule(m_standard, K):
    """Channel-count rule r* of the T1 score (r*_rel for K >= 4, else r*_PR)."""
    C = len(m_standard)
    ms = np.sort(m_standard)[::-1]
    cum = np.cumsum(ms)
    tot = cum[-1]
    r_rel = (int(np.argmax(cum / tot >= 0.95)) + 1) if tot > 1e-10 else C
    pos = ms[ms > 0]
    r_pr = (int(np.round(pos.sum() ** 2 / (pos ** 2).sum()))
            if len(pos) > 0 and (pos ** 2).sum() > 1e-20 else C)
    r_pr = max(min(r_pr, C), K - 1)
    r = r_rel if K >= 4 else r_pr
    return dict(r_star=int(max(r, K - 1)), r_rel=int(r_rel), r_pr=int(r_pr))


def channels_from_directions(W, order, k):
    """Top-k channels by summed squared loadings on the first k directions."""
    Wd = W[:, order[:max(k, 1)]]
    return np.argsort(np.sum(Wd ** 2, 1))[::-1][:k]


class ScoreRanker:
    """Caches the whitening basis so rankings for many k are cheap."""

    def __init__(self, S, y):
        self.d = direction_scores(S, y)
        self.order = {key: np.argsort(self.d[key])[::-1]
                      for key in ("standard", "jensen", "loading")}
        self.rule = count_rule(self.d["standard"], self.d["K"])
        tot = self.d["standard"][self.d["standard"] > 0].sum()
        self.loading_share = float(self.d["loading"][self.d["standard"] > 0].sum() / tot) \
            if tot > 0 else np.nan

    def channels(self, key, k):
        return channels_from_directions(self.d["W"], self.order[key], k)

    def projection(self, key, k):
        return self.d["W"][:, self.order[key][:k]]


# ---------------------------------------------------------------------------
# CSP filters and patterns (same construction as the CSP-LDA decoder)
# ---------------------------------------------------------------------------
def csp_filters(S, y, m=3):
    Sn = S / np.trace(S, axis1=1, axis2=2)[:, None, None]
    C_all = Sn.mean(0)
    C = S.shape[1]
    ev, evc = np.linalg.eigh(C_all + 1e-4 * np.eye(C))
    Ww = evc * (1. / np.sqrt(np.maximum(ev, 1e-12)))
    m = min(m, max(1, C // 2))
    filt = []
    for c in np.unique(y):
        Cw = Ww.T @ Sn[y == c].mean(0) @ Ww
        Cw = (Cw + Cw.T) / 2
        e, v = np.linalg.eigh(Cw)
        V = Ww @ v[:, np.argsort(e)[::-1]]
        filt += [V[:, :m].T, V[:, -m:].T]
    return np.vstack(filt), C_all


def csp_filter_ranking(S, y):
    Wf, _ = csp_filters(S, y)
    return np.argsort(np.sum(Wf ** 2, 0))[::-1]


def csp_pattern_ranking(S, y):
    Wf, C_all = csp_filters(S, y)
    A = C_all @ Wf.T @ np.linalg.pinv(Wf @ C_all @ Wf.T)            # (C, F) patterns
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    return np.argsort(np.sum(A ** 2, 1))[::-1]


# ---------------------------------------------------------------------------
# Barachant-Bonnet backward elimination
# ---------------------------------------------------------------------------
def _ridge(S):
    C = S.shape[-1]
    tr = np.trace(S, axis1=-2, axis2=-1)
    return S + (1e-6 * tr / C)[..., None, None] * np.eye(C)


def _batched_riemann_distance(A, B):
    """Affine-invariant distance for stacks of SPD matrices A, B (..., n, n)."""
    L = np.linalg.cholesky(B)
    Li = np.linalg.inv(L)
    M = Li @ A @ np.swapaxes(Li, -1, -2)
    ev = np.linalg.eigvalsh((M + np.swapaxes(M, -1, -2)) / 2)
    return np.sqrt(np.sum(np.log(np.maximum(ev, 1e-300)) ** 2, -1))


def bb_ranking(S, y, k_min=2):
    """Elimination order of Barachant and Bonnet (2011); returns channels from
    last-removed (most important) to first-removed. Class means are
    Riemannian means of the ridge-regularised trial covariances, computed
    once on the full montage, as in pyriemann's ElectrodeSelection."""
    from pyriemann.utils.mean import mean_riemann
    covs = _ridge(S)
    means = [mean_riemann(covs[y == c]) for c in np.unique(y)]
    pairs = [(i, j) for i in range(len(means)) for j in range(i + 1, len(means))]
    remaining = list(range(S.shape[1]))
    removed = []
    while len(remaining) > k_min:
        r = np.array(remaining)
        subs = np.array([np.delete(r, i) for i in range(len(r))])   # (c, c-1)
        score = np.zeros(len(r))
        for i, j in pairs:
            Ai = means[i][subs[:, :, None], subs[:, None, :]]
            Bj = means[j][subs[:, :, None], subs[:, None, :]]
            score += _batched_riemann_distance(Ai, Bj)
        torm = int(np.argmax(score))
        removed.append(remaining.pop(torm))
    return np.array(remaining[::-1] + removed[::-1])


# ---------------------------------------------------------------------------
# Geometric (sensorimotor) ranking
# ---------------------------------------------------------------------------
def geo_ranking(ch_names):
    import mne
    mont = mne.channels.make_standard_montage("standard_1005")
    pos = {k.lower(): v for k, v in mont.get_positions()["ch_pos"].items()}
    ref = np.stack([pos["c3"], pos["cz"], pos["c4"]])
    d = []
    for n in ch_names:
        p = pos.get(n.lower())
        d.append(np.inf if p is None else float(np.min(np.linalg.norm(ref - p, axis=1))))
    return np.argsort(np.array(d), kind="stable")


# ---------------------------------------------------------------------------
# Label-free projection
# ---------------------------------------------------------------------------
def pca_projection(S, k):
    mix = S.mean(0)
    ev, evc = np.linalg.eigh(mix)
    return evc[:, np.argsort(ev)[::-1][:k]]
