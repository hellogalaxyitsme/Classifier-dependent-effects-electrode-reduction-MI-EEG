# Data Access

Raw recordings are not distributed in this repository. All twelve datasets are
public and are downloaded through MOABB, which stores them in the MNE data
directory (`MNE_DATA`, default `$P/mne_data` after `source code/env.sh`).

## Development datasets

| Dataset | MOABB class | Cache name |
|---|---|---|
| Cho 2017 | `Cho2017` | `cho2017` |
| Lee 2019 | `Lee2019_MI` | `lee2019_mi`; both sessions: `lee2019_2s` |
| HGD (Schirrmeister 2017) | `Schirrmeister2017` | `schirrmeister2017`; original runs: `hgd_runs` |
| BNCI 2014-001 | `BNCI2014_001` | `bnci2014_001` |

The within-session caches of these datasets are computed from epoch files
extracted with MOABB 1.5.0 (`MotorImagery` paradigm, default epoch interval,
native sampling rate). Each dataset has one pickle holding, per participant,
the epochs `X`, the labels `y` and, for BNCI 2014-001, the `session` and `run`
of each trial, and a `<dataset>_channel_info.json` with the channel names.
These epoch files are not distributed; `code/scripts/build_covs_dev.py` reads
them from the directory `DEV_EPOCHS`.
The cross-session caches `lee2019_2s` and `hgd_runs` are built from the raw
data with `code/scripts/build_covs_extra.py`.

## Confirmatory datasets

| Dataset | MOABB class | Cache name |
|---|---|---|
| Dreyer 2023 | `Dreyer2023` | `dreyer2023` |
| PhysioNet MI | `PhysionetMI` | `physionetmi` |
| Yang 2025 | `Yang2025` | `yang2025` |
| Stieger 2021 | `Stieger2021` | `stieger2021` |

## Additional datasets

| Dataset | MOABB class | Cache name |
|---|---|---|
| Brandl 2020 | `Brandl2020` | `brandl2020` |
| Guttmann-Flury 2025 | `GuttmannFlury2025_MI` | `guttmannflury2025_mi` |
| Weibo 2014 | `Weibo2014` | `weibo2014` |
| Grosse-Wentrup 2009 | `GrosseWentrup2009` | `grossewentrup2009` |
| Chang 2025 (not evaluable) | `Chang2025` | `chang2025` |

## Local layout

```text
$P/mne_data/                         MOABB downloads
$P/dev_epochs/                       MOABB 1.5.0 epoch files of the development datasets
$P/cache/covs/<cache name>/          per-trial covariances, one subj_<s>.npz per participant
$P/runs/<experiment>/<cache name>/   fold-level run files
$P/results/                          aggregate outputs
```

Relevant scripts:

```text
code/scripts/download_data.py
code/scripts/build_covs_dev.py
code/scripts/build_covs_conf.py
code/scripts/build_covs_oos.py
code/scripts/build_covs_extra.py
```
