# Results Index

Aggregate outputs are stored under `results/`. Differences are in percentage
points with unadjusted 90 % confidence intervals unless a file states
otherwise.

## Hypothesis tests

- `confirmatory_tests.csv`: hypotheses H1-H6, frozen code on the stored data.
- `confirmatory_tests_rescaled.csv`: the same analysis on unit-normalised
  covariances.
- `oos_tests.csv`, `oos_pooled.csv`: addendum hypotheses O1-O6 per dataset and
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
| Table 1 | `participant_level/` (selector `FULL`) |
| Figure 1 | `followup_fluctuation_channels.csv`; selection frequencies from the `CONF_normalised` selection files |
| Figure 2, Table S8 | `contrasts_per_dataset.csv`, `contrasts_pooled.csv` |
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
| Tables S9, S10 | `followup_augmentation_size.csv`, `augmentation_correlated.csv` |
| Table S11 | `cross_session_nested.csv`, `followup_experiments.csv` (`HSPLIT`) |
| Section S3 | `floor_check.csv`, `floor_check_csplda.csv` |
| Section S6 | `followup_dist.csv`, `followup_gapshare.csv`, `followup_fluctuation_pooled.csv` |
| Section S8 | `development/` |
| Section S9 | `followup_experiments.csv`, `followup_stability.csv`, `cross_session_nested*.csv`, `spd_check.csv` |
