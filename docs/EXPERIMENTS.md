# Experiments

Every experiment writes one gzip CSV per participant and split seed to
`runs/<code>/<dataset>/subj_<s>_seed_<seed>.csv.gz` (one row per fold,
configuration and decoder, with the number of correct test trials), together
with `sel_*` files (selected channels) and `folds_*` files (fold-level
quantities such as the count r*). Existing files are skipped, so interrupted
runs can be resumed.

All protocols use within-session stratified five-fold cross-validation, and
every ranking, count, detector and decoder is fitted on the training partition
only.

## Development experiments (`run_mech.py`)

Development datasets, split seeds 42-46 unless stated.

| Code | Content |
|---|---|
| `A` | Matched-count panel at k = r* and k in {8, 16, 32}: rankings T1, JG, LO, SH, CSPF, CSPP, BB, GEO, projections PT1, PJG, PPCA, ten random subsets, full montage; decoders MDM, MDM-OAS, TS-LR, CSP-LDA |
| `A2` | Variance-fluctuation (NS) and mean log-variance (VAR) rankings |
| `A3` | Precision ranking (PREC) and its variance-scaled form (VIF) |
| `A4` | FgMDM on T1, PREC, SH, BB, CSPF and GEO |
| `B` | Accuracy against channel count on a grid from 2 to C (seed 42) |
| `C` | Training-set size: 25 % and 50 % subsamples of the training partition, three draws (seed 42) |
| `D` | Montage density: the 22 BNCI 2014-001 positions and the 19-channel 10-20 set, with selection re-run inside them |
| `E` | Selection fitted on all trials of the participant, as in non-nested pipelines (seed 42) |
| `G` | Class-independent electrodes appended to the 22-position montage (stationary or trial-varying variance) |
| `H` | Common-average reference |
| `W` | 1-s and 2-s windows (caches `<dataset>_w1`, `<dataset>_w2`) |

## Confirmatory protocol

| Code | Script | Content |
|---|---|---|
| `CONF` | `run_mech.py CONF` (frozen code) | Hypotheses H1-H6 on Dreyer 2023, PhysioNet MI, Yang 2025 and Stieger 2021; split seeds 42-44; selectors T1, SH, PREC, BB, CSPF, GEO, random subsets, M22; decoders MDM, MDM-OAS, FgMDM, TS-LR, CSP-LDA |
| `rescaled/CONF` | `run_conf_rescaled.py` | The frozen protocol, unchanged, on covariances divided by one constant per participant |
| `CONF_CSPNORM` | `run_conf_csp.py` | CSP-LDA on the confirmatory folds and subsets with training-normalised covariances |

## Follow-up experiments (`run_gaps.py`)

Covariances of each fold are divided by the mean training-set channel variance.

| Code | Content |
|---|---|
| `BADCH` | PREP-style bad-channel detection (deviation, correlation and RANSAC predictability) against PREC, NS, GEO and random deletion of the same number of electrodes |
| `KSEL` | Channel count chosen by nested five-fold cross-validation (separately for MDM and TS-LR) and fixed fractions of the montage |
| `XSESS` | Transfer across sessions, without adaptation and after unsupervised re-centring (Lee 2019, Yang 2025, Stieger 2021, BNCI 2014-001) |
| `HSPLIT` | HGD on its original training and test runs |
| `GSIZE` | Class-independent electrodes crossed with training-set size |

## Additional datasets (`run_oos.py`, frozen addendum code)

| Code | Content |
|---|---|
| `OOS` | Hypotheses O1-O6: the `KSEL` protocol with PREC, M22 and the count r* on Brandl 2020, Guttmann-Flury 2025, Weibo 2014 and Grosse-Wentrup 2009 |

## Transfer and mixing (`run_transfer_mixing.py`)

| Code | Content |
|---|---|
| `XNEST` | Transfer across sessions with the PREC count chosen by nested cross-validation in the training session |
| `XNESTL` | `XNEST` with the diagonal loading of MDM added before re-centring (sensitivity analysis including all participants) |
| `GCOR` | 32 class-independent electrodes appended to M22 and mixed with the recorded electrodes by a fixed matrix (independent, spread and leak conditions) |

## Simulation (`run_simulation.py`)

Monte-Carlo check of Propositions 1-3: four informative electrodes, 0 to 32
class-independent electrodes with log-variance fluctuation sigma in
{0, 0.25, 0.5, 1}, 25 to 200 training trials per class, 20 replicates.

## Abbreviations

| Name | Meaning |
|---|---|
| `T1` | Information-theoretic channel score (Supplementary Section S2) |
| `SH` | T1 computed with permuted training labels |
| `JG`, `LO` | Jensen-gap term and loading term of T1 |
| `PREC` | Largest diagonal entries of the regularised training precision matrix |
| `BB` | Riemannian backward elimination (Barachant and Bonnet, 2011) |
| `CSPF`, `CSPP` | Summed squared CSP filter weights and CSP patterns |
| `GEO` | Distance to C3, Cz and C4 |
| `M22` | Positions of the 22-electrode BNCI 2014-001 montage present in a dataset |
| `RAND` | Random subsets of equal size |
| `NS` | Smallest trial-to-trial fluctuation of the log-variance |
| `PREP_ALL` | Electrodes kept after PREP-style detection |
| `FULL` | Full montage |
| `rstar`, `innercv`, `fixed`, `frac*`, `matched` | Count rule r*, nested count, fixed count, fraction of the montage, count matched to detection |
| `TSLR_RAW` | TS-LR with the MDM loading instead of OAS shrinkage |
