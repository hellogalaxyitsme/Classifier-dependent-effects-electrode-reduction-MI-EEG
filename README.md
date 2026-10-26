# Classifier-dependent effects of electrode reduction in motor-imagery EEG decoding

This repository contains the code, the two frozen analysis plans and the
aggregate and participant-level results for a study of when electrode
reduction improves motor-imagery EEG decoding. Electrode reduction is evaluated
for four covariance-based decoders (MDM, FgMDM, TS-LR and CSP-LDA) on twelve
public motor-EEG datasets with nested within-session cross-validation.
Hypotheses formed on four development datasets were tested under analysis
plans frozen before four confirmatory and four additional datasets were
analysed.

Raw recordings and intermediate covariance caches are not distributed here.
All datasets are public and are obtained through MOABB.

## Repository Layout

```text
code/src/mech/               Covariance caches, channel rankings, decoders and summary statistics
code/scripts/                Cache builders, experiment runners, analyses and numerical checks
code/channel_info/           Channel names of every dataset in cache order
code/env.sh                  Environment variables used by the scripts
analysis_plans/              Frozen confirmatory plan and addendum, with SHA-256 hashes and frozen code
docs/                        Data access, preprocessing, experiments, analysis plans and results index
results/                     Aggregate result tables and numerical checks
results/participant_level/   Participant-level accuracies of every experiment
results/development/         Printed summaries of the development experiments
```

## Included Analyses

- Within-session comparison of full montages, data-driven channel rankings
  (information-theoretic score, its label-permuted version, precision ranking,
  Riemannian backward elimination, CSP filter weights, random subsets) and
  prior-knowledge montages for MDM, FgMDM, TS-LR and CSP-LDA.
- The confirmatory protocol (hypotheses H1 to H6) on four datasets, in the
  frozen form and on unit-normalised covariances.
- The addendum protocol (hypotheses O1 to O6) on four additional datasets.
- Channel-count rules, including nested cross-validation of the count, subset
  stability, PREP-style bad-channel detection and transfer across sessions with
  and without unsupervised re-centring.
- Decoder-by-reduction interaction contrasts, DerSimonian-Laird and
  Hartung-Knapp-Sidik-Jonkman pooling, and subset agreement between rankings.
- Simulation of Propositions 1 to 3 and covariance augmentation of real
  recordings with independent and correlated class-independent electrodes.
- Numerical checks of the normalisation floors and of the sessions excluded
  from the cross-session analyses.

## Not Included

- Raw EEG recordings and MOABB downloads.
- Per-trial covariance caches (`cache/`) and fold-level run files (`runs/`).
- The MOABB 1.5.0 epoch files of the development datasets (see
  [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md)).

## Environment

The scripts target Python 3.12 on Linux. All decoders run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
source code/env.sh
```

`code/env.sh` sets the project root `P` (default: the repository root), under
which `cache/`, `runs/` and `results/` are written, together with the MNE data
directory and single-threaded BLAS settings for parallel runs.

## Quick Checks

```bash
python -m compileall -q code
python code/scripts/run_mech.py --help
python code/scripts/run_gaps.py --help
python code/scripts/run_transfer_mixing.py --help
```

The frozen analysis code can be checked against the hashes recorded when the
plans were frozen; see [`docs/ANALYSIS_PLANS.md`](docs/ANALYSIS_PLANS.md).

## Reproducibility

Start with [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the run
order. Experiments and their codes are described in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), the frozen plans and their
verification in [`docs/ANALYSIS_PLANS.md`](docs/ANALYSIS_PLANS.md), the
preprocessing in [`docs/PREPROCESSING.md`](docs/PREPROCESSING.md), and every
result file, with the table or figure of the article it supports, in
[`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md).

## License

MIT; see [`LICENSE`](LICENSE).
