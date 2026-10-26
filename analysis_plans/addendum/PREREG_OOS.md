# Analysis plan addendum: confirmation on additional datasets

Written 24 October 2026, before any of the datasets below was downloaded or
inspected. The SHA-256 hash of this file was recorded in PREREG_OOS.sha256 at
the time of writing.

## Purpose

Two results were found in exploratory analyses after the frozen confirmatory
analysis:

1. Choosing the number of channels by nested inner cross-validation lets the
   label-free precision ranking (PREC) improve MDM. With the count rule r*,
   the gain was unreliable.
2. A fixed 22-electrode central montage (M22) costs the tangent-space
   classifier (TS-LR) little.

They are tested here on datasets not used before in this study.

## Datasets

Inclusion rule: every MOABB 1.7.2 motor-imagery dataset that meets all of the
following.

- Left-hand and right-hand imagery classes.
- Healthy participants.
- At least 59 EEG channels.
- At least 10 participants.
- Fixed-length, cue-based trials.
- At least one session with 20 or more trials per class.
- Not used before in this study.

Datasets meeting the rule:

| Dataset | Participants | EEG channels | Notes |
|---|---|---|---|
| Brandl2020 | 16 | 63 | One session, seven runs |
| Chang2025 | 28 | 59 | Motor-imagery paradigm, four sessions |
| GuttmannFlury2025_MI | 31 | 62 | Three sessions, 20 trials per class each |
| Weibo2014 | 10 | 60 | Left and right hand only |
| GrosseWentrup2009 | 10 | 128 | Distributed with a common average reference |

Excluded under the rule:

- Wang2026: online cursor control with variable-length trials.
- Forenzo2023: continuous 60-s pursuit trials.
- Kumar2024 (22 channels), Zhou2020 (26 or 41), Shin2017A (30) and
  HefmiIch2025 (32): too few channels.
- Liu2024 and HefmiIch2025: patient populations.
- Wu2020 (6 participants) and Beetl2021: too few participants.

A dataset that cannot be downloaded or read is reported as unavailable and is
not replaced.

## Protocol

The protocol is the KSEL protocol of code/run_gaps.py, unchanged.

- Epochs come from the MOABB MotorImagery paradigm (left and right hand,
  0.5–45 Hz pre-filter, native rate, default interval). Each trial
  covariance is computed in the 8–30 Hz band (mech/data.py).
- Within-session stratified 5-fold CV with split seeds 42, 43 and 44.
  Multi-session datasets use only the first session with at least 20 trials
  per class.
- In every fold, covariances are divided by the mean training channel
  variance.
- PREC is the diagonal of the regularised precision matrix of the training
  mean covariance.
- The channel count is chosen by inner 5-fold CV on the training partition,
  separately for MDM and TS-LR, from {4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64,
  96, C}.
- r* is computed as in the confirmatory protocol.
- M22 is the set of the 22 BNCI2014-001 positions present in the montage.
  It is evaluated only if at least 18 of the 22 positions are present.
- Decoders are MDM and TS-LR (OAS, tangent space, standardisation and
  logistic regression with inner-CV regularisation).
- No other configuration is analysed for this addendum.

## Hypotheses and decision rules

Accuracy differences are computed per participant (mean over folds and seeds)
in percentage points. Each test is one-sided at α = 0.05, using a paired t-test
across participants, which corresponds to the 90 % two-sided CI. Holm
correction is applied across the datasets within each hypothesis.

| ID | Hypothesis | Supported on a dataset if |
|---|---|---|
| O1 | MDM with PREC at the inner-CV count beats full-montage MDM | Holm-adjusted p < 0.05, difference > 0 |
| O2 | MDM with M22 beats full-montage MDM | Holm-adjusted p < 0.05, difference > 0 |
| O3 | TS-LR with M22 is non-inferior to full-montage TS-LR, margin 2 pp | Holm-adjusted p < 0.05 for difference > −2 |
| O4 | TS-LR with PREC at the inner-CV count is non-inferior to full-montage TS-LR, margin 2 pp | Holm-adjusted p < 0.05 for difference > −2 |
| O5 | MDM with PREC at the inner-CV count beats MDM with PREC at r* | Holm-adjusted p < 0.05, difference > 0 |
| O6 | MDM with PREC at the inner-CV count stays below full-montage TS-LR | Holm-adjusted p < 0.05, difference < 0 |

A hypothesis is called **confirmed** if it is supported on a majority of the
evaluable datasets and its DerSimonian–Laird pooled estimate lies on the
predicted side of the null value (0, or −2 for O3 and O4). Otherwise it is
called **not confirmed**.

For O3 and O4, results at a 1-pp margin are also reported, descriptively.
Every result is reported, whatever the outcome.

## Deviations

Any deviation from this plan is recorded with its date before the affected
results are inspected.
