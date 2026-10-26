"""Cache builder for the out-of-sample datasets (PREREG_OOS.md).

usage: python build_covs_oos.py <MoabbDatasetClass> [n_workers]

Runs build_covs_conf.work unchanged, with per-subject error handling, so that a
failed download or an empty recording of one subject does not stop the
others. Existing caches are skipped, so the builder can be re-run to retry
failed subjects.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_covs_conf as bc  # noqa: E402

_work = bc.work


def safe_work(s):
    try:
        return _work(s)
    except Exception:
        return s, "FAILED " + traceback.format_exc().strip().splitlines()[-1][:300]


bc.work = safe_work

if __name__ == "__main__":
    bc.main()
