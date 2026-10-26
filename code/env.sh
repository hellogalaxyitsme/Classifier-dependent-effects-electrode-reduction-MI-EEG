# Environment for the analysis scripts. From the repository root: source code/env.sh
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export P="${P:-$REPO}"                              # cache/, runs/ and results/ are written under $P
export PY="${PY:-python}"
export MNE_DATA="${MNE_DATA:-$P/mne_data}"          # raw MOABB downloads
export DEV_EPOCHS="${DEV_EPOCHS:-$P/dev_epochs}"    # epoch files of the development datasets
export PYTHONPATH="$REPO/code/src:$REPO/code/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
