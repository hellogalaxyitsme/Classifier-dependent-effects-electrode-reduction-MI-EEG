"""Build per-trial covariance caches for the confirmatory datasets (MOABB 1.7.2).

usage: python build_covs_conf.py <MoabbDatasetClass> [n_workers]

Epochs are extracted with the MOABB MotorImagery paradigm restricted to the
left-hand and right-hand classes, the dataset's default epoch interval, the
native sampling rate and a 0.5-45 Hz pre-filter; the 8-30 Hz band-pass and
covariance computation of mech.data are then applied. Multi-session
datasets are loaded one session at a time to bound memory. Only covariances
are computed here; no decoder is run.
"""
import json
import multiprocessing as mp
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from mech.data import CHINFO, COVDIR, trial_covs  # noqa: E402

NAME = None


def _paradigm():
    from moabb.paradigms import MotorImagery
    return MotorImagery(events=["left_hand", "right_hand"], n_classes=2, fmin=0.5, fmax=45.,
                        resample=None)


def work(s):
    import mne
    import moabb
    import moabb.datasets as md
    mne.set_log_level("ERROR")
    if NAME in ("Stieger2021", "Yang2025"):
        # raw files were fetched from the upstream host; do not re-fetch them
        moabb.set_download_provider("upstream")
    out_f = COVDIR / NAME.lower() / f"subj_{s}.npz"
    if out_f.exists():
        return s, "exists"
    par = _paradigm()
    ds = getattr(md, NAME)()
    sess_list = [None]
    if NAME == "Stieger2021":
        import glob
        import os
        root = os.path.join(os.environ["MNE_DATA"], "MNE-Stieger2021-data")
        sess_list = sorted(int(f.split("_Session_")[1].split(".")[0])
                           for f in glob.glob(os.path.join(root, f"S{s}_Session_*.mat")))
    S_all, y_all, ses_all, run_all, T, fs, names = [], [], [], [], None, None, None
    for k in sess_list:
        if k is not None:
            ds = md.Stieger2021(sessions=[k])
        epo, y, meta = par.get_data(dataset=ds, subjects=[s], return_epochs=True)
        if len(y) == 0:
            continue
        fs = float(epo.info["sfreq"])
        X = epo.get_data(copy=False)
        if names is None:
            names = list(epo.ch_names)
        elif list(epo.ch_names) != names:
            return s, f"channel mismatch in session {k}"
        T = X.shape[2]
        S_all.append(trial_covs(X, fs))
        y_all.append(np.array([{"left_hand": 0, "right_hand": 1}[v] for v in y], dtype=np.int64))
        ses_all.append(meta["session"].astype(str).values)
        run_all.append(meta["run"].astype(str).values)
    if not S_all:
        return s, "no trials"
    np.savez(out_f.with_name(out_f.stem + ".tmp.npz"), S=np.concatenate(S_all), y=np.concatenate(y_all),
             T=np.int64(T), session=np.concatenate(ses_all), run=np.concatenate(run_all), fs=fs,
             ch_names=np.array(names))
    out_f.with_name(out_f.stem + ".tmp.npz").replace(out_f)
    return s, f"n={sum(len(v) for v in y_all)} C={len(names)} T={T} fs={fs} sessions={len(S_all)}"


def main():
    global NAME
    NAME = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    import moabb.datasets as md
    subs = getattr(md, NAME)().subject_list
    if len(sys.argv) > 3:
        subs = [int(v) for v in sys.argv[3].split(",")]
    (COVDIR / NAME.lower()).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(n, maxtasksperchild=1) as pool:
        for s, st in pool.imap_unordered(work, subs):
            print(f"{time.strftime('%F %T')} {NAME} subject {s}: {st} ({time.time() - t0:.0f}s)", flush=True)
    names = None
    for f in sorted((COVDIR / NAME.lower()).glob("subj_*.npz")):
        names = [str(v) for v in np.load(f)["ch_names"]]
        break
    if names:
        CHINFO.mkdir(parents=True, exist_ok=True)
        json.dump(dict(ch_names=names, source="MOABB 1.7.2 epochs"),
                  open(CHINFO / f"{NAME.lower()}.json", "w"), indent=1)
    print(f"{NAME} DONE", flush=True)


if __name__ == "__main__":
    main()
