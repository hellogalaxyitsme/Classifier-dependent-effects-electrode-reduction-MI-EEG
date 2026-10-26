"""Confirmatory protocol re-run on unit-normalised covariances (documented deviation).

The frozen spatial score floors whitening eigenvalues and whitened class
variances at an absolute 1e-12. The confirmatory datasets are stored in V^2
(median channel variance 5e-18 to 3e-10), where that floor binds and changes
the count rule r* and, slightly, the score's ranking. The development
datasets (in uV^2) are unaffected. This script runs the frozen code
unchanged on covariances divided by one constant per subject: the mean
channel variance over all of the subject's trials. That is a change of units
to which every selector and decoder is invariant apart from absolute floors.

usage: python run_conf_rescaled.py <dataset> [--workers N]
Output: runs/rescaled/CONF/<dataset>/
"""
import sys
from pathlib import Path

import numpy as np

FROZEN = Path(__file__).resolve().parents[2] / "analysis_plans" / "confirmatory" / "code"
sys.path.insert(0, str(FROZEN))
import run_mech as rm  # noqa: E402  (frozen module, imported read-only)
from mech import data as md  # noqa: E402

_load = md.load_subject


def load_scaled(dataset, subject, session="first"):
    S_all = _load(dataset, subject, session=None)["S"]
    c = float(np.trace(S_all, axis1=1, axis2=2).mean() / S_all.shape[1])
    out = _load(dataset, subject, session=session)
    out["S"] = out["S"] / c
    return out


# module-level patches so that spawned workers apply them as well
rm.load_subject = load_scaled
rm.RUNS = md.PROJ / "runs" / "rescaled"

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "CONF"] + sys.argv[1:]
    rm.main()
