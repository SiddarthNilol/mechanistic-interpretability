#!/usr/bin/env bash
# Lab PC helper: copy fine-tuned checkpoints from HPC, serve one to the robot, or replay a recording.
# Run on lair-pc. Every command takes flags; defaults are listed below.
#
#   bash robot.sh copy   --run unhook_selected --step 3000 [--name NAME]
#   bash robot.sh serve  --name NAME [--task "unhook ring"]
#   bash robot.sh replay --name NAME --recording REC.rrd [--out OUT.rrd]
#   bash robot.sh port                       what is using the server port
#
# NAME is the folder the checkpoint is copied to, and also the policy ID: the server answers with
# its checkpoint folder's name, and the robot app only runs chunks whose policy ID matches its own.
# So set the app's Policy ID to NAME. It defaults to RUN-STEP, such as unhook_selected-3000.
#
# The robot reaches lair-pc at ws://128.238.177.201:18090/ws (not the Tailscale or 192.168 address).
# "Inference policy warmup failed: Server disconnected" on the robot and "expected GET; got POST" on
# the server are harmless; inference starts when you press start on the robot.
#
# Flags (all commands):
#   --run RUN          run folder on HPC, such as unhook_selected
#   --step STEP        checkpoint step, such as 1000 or 3000
#   --name NAME        local folder name and policy ID (default RUN-STEP)
#   --task TEXT        prompt for every request (default "unhook ring")
#   --port PORT        server port (default 18090)
#   --src DIR          run folders on HPC (default /scratch/sk12590/mechanistic-interpretability/finetune)
#   --dest DIR         local models folder (default ~/projects/policy-inference-spec/models)
#   --hpc USER@HOST    HPC login (default sk12590@login.torch.hpc.nyu.edu)
#   --recording FILE   replay: input recording
#   --out FILE         replay: output (default NAME-<recording name>.rrd next to the recording)
#   --samples N        replay: number of predictions (default 250, one per second)
#   --delta            serve a delta-actions checkpoint (adds the robot's state back; see below)
set -euo pipefail

PIS=$HOME/projects/policy-inference-spec
SERVER_PY=$HOME/projects/lair-openpi/.venv/bin/python  # has openpi and JAX
REPLAY_PY=$PIS/.venv/bin/python                        # has rerun-sdk[catalog]

RUN="" STEP="" NAME="" TASK="unhook ring" PORT=18090
SRC=/scratch/sk12590/mechanistic-interpretability/finetune
DEST=$PIS/models
HPC=sk12590@login.torch.hpc.nyu.edu
RECORDING="" OUT="" SAMPLES=250 DELTA=0

stop() { echo "STOP: $*" >&2; exit 1; }

CMD=${1:-}
[ $# -gt 0 ] && shift
while [ $# -gt 0 ]; do
    case $1 in
        --run) RUN=$2; shift 2 ;;
        --step) STEP=$2; shift 2 ;;
        --name) NAME=$2; shift 2 ;;
        --task) TASK=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --src) SRC=$2; shift 2 ;;
        --dest) DEST=$2; shift 2 ;;
        --hpc) HPC=$2; shift 2 ;;
        --recording) RECORDING=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --samples) SAMPLES=$2; shift 2 ;;
        --delta) DELTA=1; shift ;;
        *) stop "unknown flag '$1'" ;;
    esac
done
if [ -z "$NAME" ] && [ -n "$RUN" ] && [ -n "$STEP" ]; then NAME=$RUN-$STEP; fi
CKPT=$DEST/$NAME

port_busy() { ss -ltn | grep -q ":$PORT "; }

need_checkpoint() {
    [ -n "$NAME" ] || stop "give --name (or --run and --step)"
    [ -d "$CKPT/params" ] || stop "$CKPT/params doesn't exist; copy the checkpoint first"
    local n
    n=$(ls "$CKPT"/assets/*/norm_stats.json 2>/dev/null | wc -l)
    [ "$n" = 1 ] || stop "$CKPT needs exactly one assets/*/norm_stats.json, found $n"
}

server_command() {
    # The fine-tuning trains with RTC (finetune_job.sh passes --rtc-train), so the server must too.
    if [ "$DELTA" = 1 ]; then
        # server_openpi sends the model's output as joint positions, so a delta checkpoint would
        # drive the robot toward the zero pose. serve_delta.py adds the state back.
        echo "$SERVER_PY $HOME/projects/ultra-tools/data/hri-handoff/serve_delta.py"
    else
        echo "$SERVER_PY -m server.server_openpi"
    fi
}

cmd_copy() {
    [ -n "$RUN" ] && [ -n "$STEP" ] || stop "give --run and --step"
    mkdir -p "$DEST"
    local free
    free=$(df -BG --output=avail "$DEST" | tail -n 1 | tr -dc 0-9)
    echo "free in $DEST: ${free} GB; one checkpoint is about 12 GB"
    [ "$free" -ge 17 ] || stop "not enough disk space"
    echo "copying $HPC:$SRC/$RUN/$STEP -> $CKPT (HPC password and Duo next)"
    rsync -a --info=progress2 "$HPC:$SRC/$RUN/$STEP/" "$CKPT/"
    need_checkpoint
    echo "done. Set the robot app's Policy ID to: $NAME"
}

cmd_serve() {
    need_checkpoint
    port_busy && stop "port $PORT is in use (bash robot.sh port shows by what)"
    echo "serving $CKPT; the robot app's Policy ID must be: $NAME"
    echo "robot Server URL: ws://128.238.177.201:$PORT/ws"
    cd "$PIS"
    # shellcheck disable=SC2046
    PYTHONPATH=. $(server_command) --checkpoint-path "$CKPT" --rtc-train \
        --host 0.0.0.0 --port "$PORT" --task "$TASK"
}

cmd_replay() {
    need_checkpoint
    [ -f "$RECORDING" ] || stop "give --recording, an existing .rrd file"
    [ -n "$OUT" ] || OUT=$(dirname "$RECORDING")/$NAME-$(basename "$RECORDING")
    port_busy && stop "port $PORT is in use (bash robot.sh port shows by what)"
    local log=${OUT%.rrd}.server.log
    cd "$PIS"
    echo "starting the server, log $log"
    # shellcheck disable=SC2046
    PYTHONPATH=. $(server_command) --checkpoint-path "$CKPT" --rtc-train \
        --host 127.0.0.1 --port "$PORT" --task "$TASK" > "$log" 2>&1 &
    local pid=$!
    trap 'kill $pid 2>/dev/null' EXIT
    until grep -q "server listening" "$log"; do
        kill -0 "$pid" 2>/dev/null || { tail -n 20 "$log"; stop "the server stopped before listening"; }
        sleep 5
    done
    echo "replaying $RECORDING -> $OUT"
    PYTHONPATH=. "$REPLAY_PY" policy_inference_spec/replay_rrd.py --recording-path "$RECORDING" \
        --output-path "$OUT" --predict-url "ws://127.0.0.1:$PORT/ws" --policy-id "$NAME" \
        --task "$TASK" --max-samples "$SAMPLES"
    echo "done: $OUT (on the Mac: rerun --port 9878 $(basename "$OUT"))"
}

cmd_port() {
    if ss -ltnp | grep ":$PORT "; then
        echo "port $PORT is in use (if no program is named, try: sudo ss -ltnp | grep $PORT)"
    else
        echo "port $PORT is free"
    fi
}

case $CMD in
    copy | serve | replay | port) "cmd_$CMD" ;;
    *) sed -n '2,32p' "$0"; exit 1 ;;
esac
