# Preprocessing Summary

All analyses operate on per-trial spatial covariance matrices. Because the
band-pass filter acts on each channel separately, the covariance of any channel
subset is the corresponding sub-block of the full-montage covariance, so every
montage, ranking and decoder is evaluated from one cache per participant
(`code/src/mech/data.py`).

## Epochs

- MOABB `MotorImagery` paradigm, left-hand versus right-hand imagery (HGD:
  executed left hand, right hand, feet and rest; BNCI 2014-001: four-class
  imagery).
- Default epoch interval of each dataset, native sampling rate, recorded
  reference (Grosse-Wentrup 2009 is distributed with a common average
  reference).
- Confirmatory and additional datasets: MOABB 1.7.2 with a 0.5-45 Hz
  pre-filter. Development datasets: epochs extracted with MOABB 1.5.0.

## Covariances

- Zero-phase fourth-order Butterworth band-pass, 8-30 Hz, in float64.
- Per-trial covariance of the centred, filtered trial, divided by the number
  of samples.
- Normalisation by a positive constant that only changes the unit: the mean
  channel variance over all trials of a participant in the unit-normalised
  confirmatory analysis (`run_conf_rescaled.py`), and the mean training-set
  channel variance of each fold in all later analyses.

## Sessions and exclusions

- PhysioNet MI: three participants recorded at 128 Hz are excluded, leaving 106.
- Yang 2025 and Stieger 2021: every session with at least 20 trials per class in
  the confirmatory protocol; the channel-count and bad-channel analyses use the
  first three such sessions of Stieger 2021.
- Additional datasets: the first session with at least 20 trials per class.
- Cross-session analyses: participants whose session mean could not be computed
  for re-centring are listed in [`ANALYSIS_PLANS.md`](ANALYSIS_PLANS.md) and
  included in a sensitivity analysis (experiment `XNESTL`).

## Decoders

Implemented in `code/src/mech/decoders.py`.

- MDM: affine-invariant minimum distance to the class means, diagonal loading
  of 1e-6 of the mean channel variance.
- MDM-OAS: MDM on OAS-shrunk covariances.
- FgMDM: MDM after Fisher geodesic filtering in the tangent space.
- TS-LR: OAS shrinkage, tangent space at the Riemannian mean of the training
  trials, feature standardisation and l2-regularised logistic regression with
  the inverse penalty chosen from `logspace(-4, 1, 6)` by five-fold inner
  cross-validation; TS-LR with the MDM loading instead of OAS is `TSLR_RAW`.
  Inside the nested channel-count search the inverse penalty is fixed at 1.
- CSP-LDA: three CSP filter pairs per class, log-variance features and
  shrinkage linear discriminant analysis.
