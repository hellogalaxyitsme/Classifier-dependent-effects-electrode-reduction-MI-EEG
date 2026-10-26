# Analysis Plans

The study used two analysis plans, each frozen together with its analysis code
before the data it concerns were analysed. The plans were frozen internally
and recorded with SHA-256 hashes; they were not registered publicly.

| Folder | Plan | Frozen | Data analysed afterwards |
|---|---|---|---|
| `analysis_plans/confirmatory/` | `PREREGISTRATION.md`, hypotheses H1-H6 | 22 October 2026 | Dreyer 2023, PhysioNet MI, Yang 2025, Stieger 2021 |
| `analysis_plans/addendum/` | `PREREG_OOS.md`, hypotheses O1-O6 | 24 October 2026 | Brandl 2020, Guttmann-Flury 2025, Weibo 2014, Grosse-Wentrup 2009, Chang 2025 |

Each folder holds the plan, the hash file written when it was frozen and the
frozen code. `code/scripts/run_mech.py`, `analyze_conf.py`, `run_oos.py` and
`analyze_oos.py` are identical to these frozen copies.

## Verifying the hashes

```bash
cd analysis_plans/confirmatory && sha256sum -c PREREGISTRATION.sha256
cd analysis_plans/addendum && sha256sum -c PREREG_OOS.sha256
```

The frozen analysis code verifies: `analyze_conf.py` and `mech/analysis.py`
of the confirmatory plan, and `run_oos.py` and `analyze_oos.py` of the
addendum. The following files differ from the recorded hashes:

- `PREREGISTRATION.md` and `PREREG_OOS.md`: edited for wording in this release.
  References to local working files and to the computer used were removed,
  other references were reworded, the one-participant execution check of the runner is
  described in plain terms, and the opening sentence of `PREREGISTRATION.md`
  now refers to the confirmatory analyses, since that check had run the code
  on one participant before freezing. Hypotheses, decision rules, datasets,
  protocols and deviation policies are unchanged.
- `confirmatory/code/run_mech.py`, `confirmatory/code/mech/decoders.py` and
  `confirmatory/code/mech/selectors.py`: comments and docstrings were reworded in
  this release. No other line changed.
- `confirmatory/code/mech/data.py`: the default project and epoch-file
  locations were replaced by configurable paths, and comments were reworded.
  The computations are unchanged.
- `addendum/code/queue_oos.sh` (the job-queue script) is not included. It was
  changed after freezing to repeat interrupted dataset downloads, which
  affected data retrieval only.

## Deviations

- Data units. Absolute numerical floors in the frozen code made the count rule
  r* depend on the unit in which the data were stored (volts in the
  confirmatory, microvolts in the development datasets). The frozen code was
  therefore also run, unchanged, on covariances divided by one constant per
  participant (`run_conf_rescaled.py`, `analyze_conf_rescaled.py`), and CSP-LDA
  was re-evaluated on normalised inputs (`run_conf_csp.py`). Both analyses are
  reported. After normalisation the floors left the channel selections
  unchanged and had negligible effects on CSP-LDA accuracy
  (`check_numerics.py floors`, `check_numerics.py cspfloor`).
- Addendum datasets. Chang 2025 had too few trials per session to be
  evaluated, and M22 could not be defined for Grosse-Wentrup 2009, whose
  channels carry numerical labels.

Analyses outside the two plans (decoder-by-reduction interaction,
matched-estimator comparison, correlated augmentation, transfer with a nested
count) are exploratory.

## Participants excluded from the cross-session analyses

Lee 2019: 27, 50; Yang 2025: 3, 12, 18, 30, 35, 44; Stieger 2021: 25, 60
(MOABB subject numbers). In one session of each, some trial covariances are
numerically rank-deficient, so the Riemannian session mean used for
re-centring cannot be computed without regularisation (`results/spd_check.csv`).
Experiment `XNESTL` includes them with the diagonal loading of MDM added before
re-centring (`results/cross_session_nested_loaded.csv`).
