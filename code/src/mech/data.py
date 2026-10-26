"""Datasets and per-trial covariance caches.

Every analysis in this study operates on per-trial spatial covariance
matrices of 8-30 Hz band-passed EEG. Because band-pass filtering acts on each
channel separately, the covariance of any channel subset is the corresponding
sub-block of the full-montage covariance, so channel subsets, projections and
decoders can all be evaluated from one cache per subject.

Filtering: zero-phase
4th-order Butterworth band-pass at the native sampling rate (order reduced
only for very short epochs), float64 arithmetic. Per-trial covariances are
centred and divided by the number of samples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import signal

PROJ = Path(os.environ.get("P", Path(__file__).resolve().parents[3]))
# epoch files of the development datasets (extracted with MOABB 1.5.0)
DEV_EPOCHS = Path(os.environ.get("DEV_EPOCHS", PROJ / "dev_epochs"))
COVDIR = PROJ / "cache" / "covs"
CHINFO = Path(__file__).resolve().parents[2] / "channel_info"

# Development datasets; epochs extracted with MOABB 1.5.0 and read here
# unchanged.
DEV = {
    "cho2017": dict(src=DEV_EPOCHS / "cho2017_pooled.pkl",
                    fs=250., K=2),
    "lee2019_mi": dict(src=DEV_EPOCHS / "lee2019_mi_pooled.pkl",
                       fs=1000., K=2),
    "schirrmeister2017": dict(src=DEV_EPOCHS / "schirrmeister2017_pooled.pkl", fs=500., K=4),
    "bnci2014_001": dict(src=DEV_EPOCHS / "bnci2014_001_sessions.pkl",
                         fs=250., K=4, sessions=True),
}

BAND = (8., 30.)


def bandpass(X: np.ndarray, flo: float, fhi: float, sfreq: float) -> np.ndarray:
    nyq = sfreq / 2.
    lo, hi = flo / nyq, min(fhi / nyq, 0.99)
    n = X.shape[-1]
    order = 4
    while order >= 1 and 6 * order >= n:
        order -= 1
    b, a = signal.butter(order, [lo, hi], btype="band")
    return signal.filtfilt(b, a, X.astype(np.float64), axis=-1)


def trial_covs(X: np.ndarray, sfreq: float, band=BAND, chunk: int = 64) -> np.ndarray:
    """Centred per-trial covariances (divided by n_samples) of band-passed trials."""
    n, C, T = X.shape
    out = np.empty((n, C, C), dtype=np.float64)
    for i in range(0, n, chunk):
        Xf = bandpass(X[i:i + chunk], band[0], band[1], sfreq)
        Xf -= Xf.mean(-1, keepdims=True)
        out[i:i + chunk] = np.einsum("nct,ndt->ncd", Xf, Xf) / T
    return out


def cov_path(dataset: str, subject) -> Path:
    return COVDIR / dataset / f"subj_{subject}.npz"


def load_subject(dataset: str, subject, session: str | None = "first"):
    """Return dict(S, y, T, session) for one subject.

    session="first" keeps the first recording session (within-session
    analyses of multi-session datasets); None keeps all trials.
    """
    d = np.load(cov_path(dataset, subject), allow_pickle=True)    # caches written by this project
    S, y, T = d["S"], d["y"], int(d["T"])
    sess = d["session"].astype(str) if "session" in d.files else np.array(["0"] * len(y))
    if session == "first":
        m = sess == sorted(set(sess.tolist()))[0]
        S, y, sess = S[m], y[m], sess[m]
    elif session is not None:
        m = sess == session
        S, y, sess = S[m], y[m], sess[m]
    return dict(S=S, y=y, T=T, session=sess)


def subjects(dataset: str):
    p = COVDIR / dataset
    return sorted(int(f.stem.split("_")[1]) for f in p.glob("subj_*.npz"))


def channel_names(dataset: str):
    f = CHINFO / f"{dataset}.json"
    with open(f) as fh:
        return json.load(fh)["ch_names"]
