# Scripts

| Script | Purpose |
|---|---|
| `download_data.py` | Download one MOABB dataset |
| `build_covs_dev.py` | Covariance caches of the development datasets from their epoch files |
| `build_covs_conf.py` | Covariance caches of the confirmatory datasets |
| `build_covs_oos.py` | Covariance caches of the additional datasets |
| `build_covs_extra.py` | Session- and run-labelled caches of Lee 2019 and HGD |
| `run_mech.py` | Development experiments and the confirmatory protocol (frozen code) |
| `run_conf_rescaled.py` | Frozen confirmatory protocol on unit-normalised covariances |
| `run_conf_csp.py` | CSP-LDA on the confirmatory folds with normalised covariances |
| `run_gaps.py` | Follow-up experiments: bad-channel detection, channel count, cross-session transfer, augmentation |
| `run_oos.py` | Addendum protocol on the additional datasets (frozen code) |
| `run_transfer_mixing.py` | Cross-session transfer with a nested count, loaded re-centring, correlated augmentation |
| `run_simulation.py` | Monte-Carlo check of Propositions 1-3 |
| `analyze_conf.py` | Confirmatory hypotheses H1-H6 (frozen code) |
| `analyze_conf_rescaled.py` | The frozen analysis applied to the unit-normalised run |
| `analyze_oos.py` | Addendum hypotheses O1-O6 (frozen code) |
| `analyze_gaps.py` | Summaries of the follow-up experiments |
| `analyze_contrasts.py` | Interaction contrasts, pooled estimates, subset agreement, transfer and mixing |
| `analyze_A.py`, `analyze_CDE.py`, `analyze_G.py`, `analyze_HW.py`, `analyze_stability.py` | Summaries of the development experiments |
| `check_numerics.py` | Normalisation floors and sessions excluded from the cross-session analyses |
| `export_participant_level.py` | Participant-level accuracies of every experiment |

Runners accept `--workers`, `--subjects` and `--seeds` where applicable; see
`--help`.
