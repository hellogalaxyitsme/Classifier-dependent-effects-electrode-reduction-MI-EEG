# Code Overview

```text
src/mech/        Package: covariance caches (data.py), channel rankings and projections
                 (selectors.py), decoders (decoders.py) and participant-level summaries (analysis.py)
scripts/         Command-line entry points
channel_info/    Channel names of every dataset in cache order (read by mech.data)
env.sh           Project root, data directories, PYTHONPATH and thread settings
```

The package is installed with `pip install -e .` from the repository root;
`source code/env.sh` additionally places `code/src` and `code/scripts` on
`PYTHONPATH`, which the frozen addendum code requires.

## Script Groups

- `download_*` and `build_covs_*`: dataset download and per-trial covariance caches.
- `run_*`: experiment runners; each writes fold-level files under `runs/<code>/<dataset>/`.
- `analyze_*`: participant-level aggregation, hypothesis tests and summary tables.
- `check_numerics.py`: numerical checks of the normalisation floors and of
  sessions whose Riemannian mean cannot be computed without regularisation.
- `export_participant_level.py`: participant-level accuracies of every experiment.

`run_mech.py`, `analyze_conf.py`, `run_oos.py` and `analyze_oos.py` are
identical to the frozen copies in `analysis_plans/`. See
[`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md) for the experiment codes and
[`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the run order.
