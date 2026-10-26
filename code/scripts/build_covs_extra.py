"""Per-trial covariance caches that keep session and run labels for the
development datasets (cross-session and original-split analyses).

usage: python build_covs_extra.py <lee2019_2s|hgd_runs> [n_workers] [subjects]

lee2019_2s  Lee2019_MI, both sessions, left/right hand (2 classes)
hgd_runs    Schirrmeister2017, original train/test runs, 4 classes

Epoching and covariances follow build_covs_conf.py (MOABB 1.7.2 MotorImagery
paradigm, native rate, 0.5-45 Hz pre-filter, then the 8-30 Hz band-pass and
covariance of mech.data). Raw files are read from the MNE data folder (MNE_DATA).
"""
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
RAW = os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
os.environ["MNE_DATA"] = RAW
for key in ("LEE2019_MI", "SCHIRRMEISTER2017"):
    os.environ[f"MNE_DATASETS_{key}_PATH"] = RAW
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from mech.data import CHINFO, COVDIR, trial_covs  # noqa: E402

SPEC = {
    "lee2019_2s": dict(cls="Lee2019_MI", events=["left_hand", "right_hand"]),
    "hgd_runs": dict(cls="Schirrmeister2017", events=["left_hand", "right_hand", "feet", "rest"]),
}
NAME = None


def work(s):
    import mne
    import moabb
    import moabb.datasets as md
    from moabb.paradigms import MotorImagery
    mne.set_log_level("ERROR")
    moabb.set_download_provider("upstream")
    spec = SPEC[NAME]
    out_f = COVDIR / NAME / f"subj_{s}.npz"
    if out_f.exists():
        return s, "exists"
    par = MotorImagery(events=spec["events"], n_classes=len(spec["events"]), fmin=0.5, fmax=45.,
                       resample=None)
    ds = getattr(md, spec["cls"])()
    epo, y, meta = par.get_data(dataset=ds, subjects=[s], return_epochs=True)
    fs = float(epo.info["sfreq"])
    X = epo.get_data(copy=False)
    lab = {e: i for i, e in enumerate(spec["events"])}
    tmp = out_f.with_name(out_f.stem + ".tmp.npz")
    np.savez(tmp, S=trial_covs(X, fs), y=np.array([lab[v] for v in y], dtype=np.int64), T=np.int64(X.shape[2]),
             session=meta["session"].astype(str).values, run=meta["run"].astype(str).values, fs=fs,
             ch_names=np.array(epo.ch_names))
    tmp.replace(out_f)
    return s, f"n={len(y)} C={X.shape[1]} T={X.shape[2]} fs={fs} sessions={sorted(set(meta['session']))}"


def main():
    global NAME
    NAME = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    import moabb.datasets as md
    subs = getattr(md, SPEC[NAME]["cls"])().subject_list
    if len(sys.argv) > 3:
        subs = [int(v) for v in sys.argv[3].split(",")]
    (COVDIR / NAME).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(n, maxtasksperchild=1) as pool:
        for s, st in pool.imap_unordered(work, subs):
            print(f"{time.strftime('%F %T')} {NAME} subject {s}: {st} ({time.time() - t0:.0f}s)", flush=True)
    for f in sorted((COVDIR / NAME).glob("subj_*.npz")):
        CHINFO.mkdir(parents=True, exist_ok=True)
        json.dump(dict(ch_names=[str(v) for v in np.load(f)["ch_names"]], source="MOABB 1.7.2 epochs"),
                  open(CHINFO / f"{NAME}.json", "w"), indent=1)
        break
    print(f"{NAME} DONE", flush=True)


if __name__ == "__main__":
    main()
