#!/bin/bash
# OPTIONAL. The jobs use the existing lair-openpi environment by default (see jobs/env.sh), so
# this is only needed to run on unmodified upstream openpi instead:
#
#   bash jobs/setup_env.sh
#   OPENPI_VENV=$PWD/third_party/openpi/.venv source jobs/env.sh
#
# Known risk: lair-openpi's pyproject drops the pynput dependency because its evdev build fails
# on HPC nodes. Upstream does not, so `uv sync` here may fail the same way.
#
# Clones openpi at the commit the code was written against and installs it the way openpi's
# README says. Nothing inside third_party/openpi is edited.
#
# Everything lands on scratch: the environment is third_party/openpi/.venv inside this
# repository (which must itself be on scratch), and uv's download cache and its Python go to
# $SCRATCH/.cache (see jobs/caches.sh). Nothing large is written to /home.
set -euo pipefail
: "${SCRATCH:?SCRATCH is not set; expected /scratch/<netid>}"
cd "$(dirname "$0")/.."
case "$(pwd -P)" in
    "$(cd "$SCRATCH" && pwd -P)"/*) ;;
    *) echo "this repository is at $(pwd -P), not under $SCRATCH: the environment would not be on scratch" >&2; exit 1 ;;
esac
source jobs/caches.sh
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479

if [ ! -d third_party/openpi/.git ]; then
    git clone https://github.com/Physical-Intelligence/openpi third_party/openpi
fi
git -C third_party/openpi fetch --quiet origin
git -C third_party/openpi checkout --quiet "$OPENPI_COMMIT"

cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
echo "openpi $(git rev-parse HEAD) installed in $PWD/.venv"
echo "caches: $SCRATCH/.cache (uv, uv-python, openpi, huggingface)"
