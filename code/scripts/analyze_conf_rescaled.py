"""Apply the frozen analyze_conf.py, unchanged, to the unit-normalised
confirmatory re-run (runs/rescaled/CONF; see run_conf_rescaled.py).

usage: CONF_OUT=<file.csv> [CONF_DENSE=...] python analyze_conf_rescaled.py
"""
import os
import runpy
import sys
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[2] / "analysis_plans" / "confirmatory" / "code"
sys.path.insert(0, str(FROZEN))
import mech.analysis as ma  # noqa: E402  (frozen module, imported read-only)

ma.RUNS = ma.PROJ / "runs" / "rescaled"
os.environ.setdefault("CONF_OUT", "confirmatory_tests_rescaled.csv")
runpy.run_path(str(FROZEN / "analyze_conf.py"), run_name="__main__")
