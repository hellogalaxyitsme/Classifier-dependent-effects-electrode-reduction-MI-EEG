# Results Index

Aggregate outputs are stored under `results/`. Differences are in percentage
points with unadjusted 90 % confidence intervals unless a file states
otherwise.

## Hypothesis tests

- `confirmatory_tests.csv`: hypotheses H1-H6, frozen code on the stored data.
- `confirmatory_tests_rescaled.csv`: the same analysis on unit-normalised
  covariances.
- `oos_tests.csv`, `oos_pooled.csv`: additional-dataset hypotheses O1-O6 per dataset and
  DerSimonian-Laird pooled estimates.

## Contrasts and pooling

- `contrasts_per_dataset.csv`: per-dataset accuracy changes, decoder-by-reduction
  interaction (including the MDM-OAS version) and reduced MDM minus
  full-montage TS-LR, with the share of participants losing more than 3 pp.
- `contrasts_pooled.csv`: DerSimonian-Laird and REML estimates with
  Hartung-Knapp-Sidik-Jonkman intervals.
- `subset_agreement.csv`: Jaccard agreement of T1 with its label-permuted
  version and with the precision ranking.

## Follow-up experiments

- `followup_experiments.csv`: bad-channel detection, channel-count rules and
  cross-session transfer (experiments `BADCH`, `KSEL`, `XSESS`, `HSPLIT`).
- `followup_dist.csv`: participant-level distribution of the accuracy changes
  on the confirmatory datasets.
- `followup_gapshare.csv`: share of the full-montage MDM-to-TS-LR difference
  recovered by reduction.
- `followup_meta.csv`: random-effects summaries across the confirmatory datasets.
- `followup_stability.csv`: Nogueira stability within and across sessions.
- `followup_fluctuation_channels.csv`, `followup_fluctuation_subjects.csv`,
  `followup_fluctuation_pooled.csv`: unique-variance fluctuation per electrode
  and participant, and its pooled associations with the accuracy changes.
- `followup_augmentation_size.csv`: independent covariance augmentation crossed
  with training-set size.

## Transfer, mixing and simulation

- `cross_session_nested.csv`: transfer across sessions with the nested count
  and with M22, without adaptation and after re-centring.
- `cross_session_nested_loaded.csv`: the same with loaded re-centring, for all
  participants and for the participants of the main analysis.
- `augmentation_correlated.csv`: correlated covariance augmentation, including
  the share of recorded electrodes kept by PREC deletion.
- `simulation_propositions.csv`: simulated and predicted accuracies for every
  replicate and condition.

## Controls and runtime

Written by `code/scripts/run_controls.py` and `code/scripts/analyze_controls.py`.
Intervals in these files are two-sided 95 % t intervals unless the column
name says otherwise.

- `random_m22_summary.csv`, `random_m22_subjects.csv`, `random_m22_raw.csv.gz`:
  M22 against random subsets of the same size on identical folds.
- `random_nested_summary.csv`, `random_nested_subjects.csv`,
  `random_nested_raw.csv.gz`: PREC with the nested count against random subsets
  of the count chosen in the same outer fold.
- `fixed_count_random.csv`: PREC (or the loading ranking LO) and GEO against
  random subsets of the same fixed size (confirmatory k = 16; development
  k-grid).
- `shared_subset_interactions.csv`: MDM-minus-TS-LR interaction when both
  decoders receive the same PREC subset (50 % of the montage, or 16 electrodes).
- `m22_tslr_participants.csv`, `m22_tslr_distribution.csv`: per-participant
  TS-LR change with M22, with proportions and quartiles.
- `bandlimited_summary.csv`, `bandlimited_sigma.csv`, `bandlimited_raw.csv.gz`:
  band-limited stationary Gaussian channels appended to M22.
- `timing_summary.csv`, `timing_raw.csv`: single-threaded training and
  inference time, and size of the tangent-space feature matrix.

## Numerical checks

- `floor_check.csv`: floors and T1 subsets under participant-level and
  training-fold normalisation, every confirmatory fold.
- `floor_check_csplda.csv`: CSP-LDA under both normalisations in the folds where
  its floor is reached.
- `spd_check.csv`: sessions of the participants excluded from the cross-session
  analyses.

## Development experiments

- `development/A_hgd.txt`, `development/B.txt` to `development/E.txt`: printed
  summaries of experiments A to E.

## Participant-level results

`participant_level/<code>_<dataset>.csv.gz` holds one row per participant and
configuration (montage or session pair, training fraction, selector, count
rule, decoder) with the accuracy in percent; `CONF_normalised_*` is the
unit-normalised confirmatory run.

## Article items

| Article item | Result files |
|---|---|
| Table 2 | `participant_level/` (selector `FULL`) |
| Table 3 | `contrasts_per_dataset.csv`, `contrasts_pooled.csv` |
| Figure 1 | `followup_fluctuation_channels.csv`; selection frequencies from the `CONF_normalised` selection files |
| Figure 2 | `contrasts_per_dataset.csv`, `contrasts_pooled.csv`, `oos_tests.csv` |
| Figure 3 | `subset_agreement.csv`, `followup_fluctuation_channels.csv`, `followup_experiments.csv` (`BADCH`) |
| Figure 4 | `simulation_propositions.csv`, `followup_augmentation_size.csv`, `augmentation_correlated.csv` |
| Figure 5 | `followup_experiments.csv` (`KSEL`), `followup_stability.csv`, `cross_session_nested.csv` |
| Table S1 | `docs/PREPROCESSING.md`, `code/channel_info/` |
| Table S2 | `subset_agreement.csv` |
| Table S3 | `confirmatory_tests.csv`, `confirmatory_tests_rescaled.csv` |
| Table S4 | `oos_tests.csv`, `oos_pooled.csv` |
| Table S5 | `contrasts_pooled.csv` |
| Table S6 | `followup_experiments.csv` (`BADCH`) |
| Table S7 | `participant_level/CONF_normalised_*`, `followup_dist.csv` |
| Table S8 | `contrasts_per_dataset.csv` |
| Table S9 | `random_m22_summary.csv`, `random_nested_summary.csv` |
| Tables S10, S11 | `followup_augmentation_size.csv`, `augmentation_correlated.csv` |
| Table S12 | `bandlimited_summary.csv` |
| Table S13 | `cross_session_nested.csv`, `followup_experiments.csv` (`HSPLIT`) |
| Table S14 | `fixed_count_random.csv` |
| Table S15 | `timing_summary.csv` |
| Figure S1 | `m22_tslr_participants.csv`, `bandlimited_summary.csv`, `followup_augmentation_size.csv` |
| Section S3 | `floor_check.csv`, `floor_check_csplda.csv` |
| Section S6 | `followup_dist.csv`, `followup_gapshare.csv`, `followup_fluctuation_pooled.csv`, `shared_subset_interactions.csv`, `m22_tslr_distribution.csv` |
| Section S8 | `development/` |
| Section S9 | `followup_experiments.csv`, `followup_stability.csv`, `cross_session_nested*.csv`, `spd_check.csv` |
