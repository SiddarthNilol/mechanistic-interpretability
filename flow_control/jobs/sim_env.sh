# Source this for the simulator side (LIBERO, Python 3.8):  source jobs/sim_env.sh
: "${SCRATCH:?SCRATCH is not set; expected /scratch/<netid>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/jobs/caches.sh"
source "${LIBERO_VENV:-$SCRATCH/libero-venv}/bin/activate"
export LIBERO_CONFIG_PATH=$SCRATCH/.cache/libero
export PYTHONPATH="$ROOT/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}  # off-screen rendering on the GPU node
export PYTHONUNBUFFERED=1
