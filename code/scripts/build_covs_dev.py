"""Build per-trial covariance caches for the development datasets.

usage: python build_covs_dev.py <dataset> [n_workers]
Reads the MOABB 1.5.0 epoch files of the development datasets (directory DEV_EPOCHS), writes
cache/covs/<dataset>/subj_<s>.npz with S (n, C, C), y, T and session labels,
and channel_info/<dataset>.json with the channel names in cache order.
"""
import json
import multiprocessing as mp
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from mech.data import CHINFO, COVDIR, DEV, DEV_EPOCHS, trial_covs  # noqa: E402

N_CH = {"cho2017": 64, "lee2019_mi": 62, "schirrmeister2017": 128, "bnci2014_001": 22}
DATA = None


WIN = None      # optional (start_s, length_s) sub-window of each epoch
OUT = None


def work(args):
    name, s = args
    v = DATA[s]
    X = np.asarray(v["X"])[:, :N_CH[name], :]
    if WIN is not None:
        fs = DEV[name]["fs"]
        a = int(round(WIN[0] * fs))
        X = X[:, :, a:a + int(round(WIN[1] * fs))]
    S = trial_covs(X, DEV[name]["fs"])
    out = dict(S=S, y=np.asarray(v["y"], dtype=np.int64), T=np.int64(X.shape[2]))
    if "session" in v:
        out["session"] = np.asarray(v["session"]).astype(str)
        out["run"] = np.asarray(v["run"]).astype(str)
    f = COVDIR / OUT / f"subj_{s}.npz"
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(f)
    return s, X.shape


def main():
    """usage: build_covs_dev.py <dataset> [n_workers] [window_start_s window_length_s]"""
    global DATA, WIN, OUT
    name = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    OUT = name
    if len(sys.argv) > 4:
        WIN = (float(sys.argv[3]), float(sys.argv[4]))
        OUT = f"{name}_w{sys.argv[4].replace('.', 'p')}"
    (COVDIR / OUT).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(DEV[name]["src"], "rb") as fh:
        DATA = pickle.load(fh)
    print(f"{name}: loaded {len(DATA)} subjects in {time.time() - t0:.0f}s", flush=True)
    src = {"cho2017": "cho2017", "lee2019_mi": "lee2019_mi", "schirrmeister2017": "schirrmeister2017",
           "bnci2014_001": "bnci2014_001"}[name]
    info = json.load(open(DEV_EPOCHS / f"{src}_channel_info.json"))
    names = info["datasets"][src]["ch_names"][:N_CH[name]]
    assert len(names) == N_CH[name], (len(names), N_CH[name])
    CHINFO.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ch_names=names, source=f"{src}_channel_info.json"),
              open(CHINFO / f"{OUT}.json", "w"), indent=1)
    todo = [(name, s) for s in DATA if not (COVDIR / OUT / f"subj_{s}.npz").exists()]
    with mp.get_context("fork").Pool(n) as pool:
        for s, shp in pool.imap_unordered(work, todo):
            print(f"{OUT} subject {s}: X={shp} ({time.time() - t0:.0f}s)", flush=True)
    print(f"{OUT} DONE", flush=True)


if __name__ == "__main__":
    main()
