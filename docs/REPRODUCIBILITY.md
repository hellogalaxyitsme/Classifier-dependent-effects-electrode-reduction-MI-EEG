# Reproducibility Guide

The workflow has five stages: obtain the datasets, build covariance caches, run
the experiments, aggregate the results and run the numerical checks. Every
stage runs on CPU. Commands are run from the repository root after
`source code/env.sh`; experiment runners take `--workers N` for parallel
participants.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
source code/env.sh
```

## 2. Datasets and covariance caches

```bash
# confirmatory and additional datasets (MOABB 1.7.2)
python code/scripts/download_data.py Dreyer2023
python code/scripts/build_covs_conf.py Dreyer2023 8      # PhysionetMI, Yang2025, Stieger2021 likewise
python code/scripts/build_covs_oos.py Brandl2020 8       # GuttmannFlury2025_MI, Weibo2014, GrosseWentrup2009, Chang2025

# development datasets (epoch files in $DEV_EPOCHS, see docs/DATA_ACCESS.md)
python code/scripts/build_covs_dev.py cho2017 8          # lee2019_mi, schirrmeister2017, bnci2014_001 likewise
python code/scripts/build_covs_extra.py lee2019_2s 8     # both Lee 2019 sessions; hgd_runs for the original HGD split
```

## 3. Experiments

```bash
# development experiments (codes in docs/EXPERIMENTS.md)
python code/scripts/run_mech.py A cho2017 --workers 8    # A, A2, A3, A4, B, C, D, E, G, H, W

# confirmatory protocol: frozen code, unit-normalised re-run and CSP-LDA on normalised inputs
python code/scripts/run_mech.py CONF dreyer2023 --workers 8
python code/scripts/run_conf_rescaled.py dreyer2023 --workers 8
python code/scripts/run_conf_csp.py dreyer2023 --workers 8

# follow-up experiments
python code/scripts/run_gaps.py KSEL physionetmi --workers 8   # BADCH, KSEL, XSESS, HSPLIT, GSIZE

# additional datasets (frozen addendum code)
python analysis_plans/addendum/code/run_oos.py OOS brandl2020 --workers 8

# transfer with a nested count, loaded re-centring, correlated augmentation, simulation
python code/scripts/run_transfer_mixing.py XNEST stieger2021 --workers 8 --seeds 42
python code/scripts/run_transfer_mixing.py XNESTL stieger2021 --workers 8 --seeds 42
python code/scripts/run_transfer_mixing.py GCOR cho2017 --workers 8
python code/scripts/run_simulation.py
```

`run_conf_rescaled.py`, `analyze_conf_rescaled.py` and `run_conf_csp.py`
import the frozen confirmatory code from `analysis_plans/confirmatory/code/`
read-only.

## 4. Aggregation

```bash
python code/scripts/analyze_conf.py                       # results/confirmatory_tests.csv
python code/scripts/analyze_conf_rescaled.py              # results/confirmatory_tests_rescaled.csv
python code/scripts/analyze_gaps.py dist stab mech exp    # results/followup_*.csv
python code/scripts/analyze_oos.py                        # results/oos_tests.csv, results/oos_pooled.csv
python code/scripts/analyze_contrasts.py                  # contrasts, pooling, subset agreement
python code/scripts/analyze_contrasts.py xnest            # also: xnestl, gcor
python code/scripts/analyze_A.py schirrmeister2017 > results/development/A_hgd.txt
python code/scripts/analyze_CDE.py B > results/development/B.txt   # C, D, E likewise
python code/scripts/export_participant_level.py
```

## 5. Numerical checks

```bash
python code/scripts/check_numerics.py floors      # normalisation floors, all confirmatory folds
python code/scripts/check_numerics.py cspfloor    # CSP-LDA in the folds where its floor is reached
python code/scripts/check_numerics.py spd         # sessions excluded from the cross-session analyses
```

The aggregate outputs of every stage are included under `results/`, so the
reported numbers can be inspected without rerunning the experiments
([`RESULTS_INDEX.md`](RESULTS_INDEX.md)).
