# Source this in every shell and job:  source jobs/env.sh
# Activates the openpi environment and points every cache at scratch (/home has very little space).
# The environment is the existing lair-openpi one ($SCRATCH/lair-openpi/.venv). To use another,
# set OPENPI_VENV first, e.g. the upstream one that jobs/setup_env.sh builds:
#   OPENPI_VENV=$PWD/third_party/openpi/.venv source jobs/env.sh
: "${SCRATCH:?SCRATCH is not set; expected /scratch/<netid>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/jobs/caches.sh"
source "${OPENPI_VENV:-$SCRATCH/lair-openpi/.venv}/bin/activate"
export PYTHONUNBUFFERED=1
