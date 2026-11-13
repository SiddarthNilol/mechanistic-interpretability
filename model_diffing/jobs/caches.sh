# Everything that would otherwise be written under /home goes to $SCRATCH/.cache instead.
# Sourced by env.sh and setup_env.sh.
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$SCRATCH/.cache/openpi}       # checkpoints (about 25 GB)
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}                    # LIBERO episode files (about 2.5 GB)
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/.cache/uv}                   # uv's package downloads (several GB)
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$SCRATCH/.cache/uv-python}  # the Python uv installs
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache}                  # pip, matplotlib fonts and others
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
mkdir -p "$SCRATCH/.cache"
