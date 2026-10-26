# Confirmatory analysis plan

Frozen before the confirmatory analyses. Prior contact with the confirmatory
data, including a one-participant execution check, is listed below. The SHA-256
of this file and the frozen code (`code/run_mech.py`, `code/mech/*.py`,
`code/analyze_conf.py`) are recorded in `PREREGISTRATION.sha256` together with
the freezing time. Any deviation will be reported as such.

## Prior contact with the confirmatory data

Raw files were downloaded with MOABB 1.7.2 and per-trial covariance caches
were computed (8–30 Hz, native rate). Only trial counts, channel names and
sampling rates were inspected. The runner was executed once on one participant
(Dreyer2023 subject 1, seed 42) to check that it completed; its accuracies were
not inspected and its output is not part of the analysis. No other decoding
was run.

## Datasets and inclusion

| Dataset | Channels | Role | Unit of analysis |
|---|---|---|---|
| PhysionetMI | 64 | dense | 106 subjects (subjects recorded at 128 Hz, 88/92/100, excluded); imagined left vs right hand, MOABB default epoch |
| Stieger2021 | 62 | dense | all sessions with ≥ 20 trials per class; left vs right hand (MOABB events), epoch [0, 3] s |
| Yang2025 | 59 | dense | all sessions with ≥ 20 trials per class; left vs right hand |
| Dreyer2023 | 27 | sparse control | single session; left vs right hand |

## Protocol (identical to the development protocol)

Within-session stratified 5-fold CV, split seeds 42, 43, 44. Selection, counts
and every decoder fitted on the outer training partition only. Selectors at
k = r* (fold-local count rule of the T1 score) and k = 16: T1 (information-theoretic
channel score),
SH (T1 with permuted training labels), PREC (diagonal of the regularised
precision matrix of the training mixture covariance), BB (Barachant–Bonnet),
CSPF (CSP filter weights), GEO (distance to C3/Cz/C4), 10 random subsets; the
fixed central montage M22 (the BNCI2014_001 positions present in the montage);
the full montage. Decoders: MDM, MDM-OAS, FgMDM, TS-LR, CSP-LDA.

Subject-level accuracy: correct over test trials pooled over folds, averaged
over random draws, sessions and split seeds within subject. All differences in
percentage points; 90 % two-sided t intervals; tests at α = 0.05.

## Hypotheses and decision rules (primary: k = r*, MDM unless stated)

- **H1 Label-free equivalence.** T1 − SH and T1 − PREC lie within ±1 pp (90 % CI inside [−1, 1]) on each dataset.
- **H2 MDM gain on dense montages.** PREC − FULL > 0 (one-sided paired t; Holm over PhysionetMI, Stieger2021, Yang2025).
- **H3 Decoder specificity.** On each dense dataset the MDM gain exceeds the TS-LR gain (paired difference of differences > 0, one-sided, Holm over the three datasets), and for TS-LR and FgMDM, PREC − FULL < +1 pp (upper 90 % bound).
- **H4 Supervision adds nothing for MDM.** PREC is non-inferior to BB and to CSPF with a 1-pp margin (lower 90 % bound of PREC − X > −1 pp) on each dense dataset.
- **H5 Fixed central montage.** M22 − FULL > 0 for MDM on each dense dataset (one-sided, Holm).
- **H6 Sparse control.** On Dreyer2023 the MDM gain PREC − FULL is smaller than on every dense dataset, and its 90 % CI lies within ±2 pp.
- **H7 Label-free selection does not leak.** Not tested on these data (development only).

Predicted magnitudes (from development data, for interpretation only): MDM
gain +3 to +8 pp on dense datasets; TS-LR gain between −2 and +1 pp.

## Secondary and exploratory analyses

- Session-pooled training size on Stieger2021 (MDM gain at 1, 2 and 4 pooled sessions); predicted not to decrease by more than 1 pp.
- Margin sensitivity (1, 2, 3, 5 pp) for the TS-LR non-inferiority of PREC.
- Per-subject distribution of the MDM gain; proportion of subjects with gain < 0.
- Stability (Nogueira) of PREC and T1 subsets across folds, seeds and sessions.

## Deviation policy

Loader failures or subjects with too few trials are excluded and listed. If a
dataset cannot be loaded at all, it is dropped and reported; no replacement is
added after results are seen.
