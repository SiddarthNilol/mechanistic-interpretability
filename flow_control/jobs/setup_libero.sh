#!/bin/bash
# Module 1: the LIBERO simulator environment (Python 3.8), following openpi's
# examples/libero/README.md ("Without Docker"). Run once, on a login node, from model_diffing:
#
#   bash jobs/setup_libero.sh
#
# Writes only under $SCRATCH: the LIBERO source in third_party/libero (not committed), at the
# commit lair-openpi pins as its submodule, the venv in $SCRATCH/libero-venv, and LIBERO's own
# config in $SCRATCH/.cache/libero (instead of ~/.libero).
set -euo pipefail
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}  # the torch 1.11 wheels are large
: "${SCRATCH:?SCRATCH is not set; expected /scratch/<netid>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/jobs/caches.sh"
OPENPI=${OPENPI_DIR:-$SCRATCH/lair-openpi}
LIBERO=$ROOT/third_party/libero
VENV=${LIBERO_VENV:-$SCRATCH/libero-venv}
COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c  # git -C $OPENPI ls-tree HEAD third_party/libero

if [ ! -d "$LIBERO/.git" ]; then
    git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO "$LIBERO"
fi
git -C "$LIBERO" checkout -q "$COMMIT"

uv venv --clear --python 3.8 "$VENV"  # replaces a half-built one from an earlier failed run
source "$VENV/bin/activate"
# evdev (keyboard input, pulled in by robosuite -> pynput for teleoperation, which we never use)
# does not build on Torch: no kernel headers. `uv pip sync` installs exactly the listed packages,
# so dropping its line leaves pynput installed without it.
REQS=$SCRATCH/.cache/libero-requirements.txt
grep -v '^evdev==' "$OPENPI/examples/libero/requirements.txt" > "$REQS"
uv pip sync "$REQS" "$LIBERO/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e "$OPENPI/packages/openpi-client"
uv pip install -e "$LIBERO"

# LIBERO asks on first import where to keep its data; answer "N" (use the defaults inside the
# source tree) once, with the config file under scratch.
export LIBERO_CONFIG_PATH=$SCRATCH/.cache/libero
export PYTHONPATH="$LIBERO${PYTHONPATH:+:$PYTHONPATH}"  # as openpi's README: LIBERO's editable install is not importable on its own
echo N | python -c "from libero.libero import benchmark; print('LIBERO tasks in libero_90:', benchmark.get_benchmark_dict()['libero_90']().n_tasks)"
echo "done: source jobs/sim_env.sh to use it"
