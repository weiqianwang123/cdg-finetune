#!/bin/bash
# One fine-tune run on the GPUs of this machine, with the openpi fork beside this repo
# (../openpi) and the data beside it too (../data, from scripts/fetch.py):
#   bash scripts/train.sh RUN [openpi train.py options]
#   bash scripts/train.sh abc_yam_v3_ft_pour_gt_diverse100 --fsdp-devices 8
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh abc_yam_v3_ft_pour_gt_diverse100 --fsdp-devices 2
#   bash scripts/train.sh abc_yam_v3_ft_pour_gt_diverse100 --fsdp-devices 8 --resume   # after an interruption
# RUN names both the config (configs/RUN.yaml) and the experiment (checkpoints/RUN/RUN/<step>/).
# Logs go to Weights & Biases when WANDB_API_KEY is set, else nowhere but the console.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?the run's name (a config in $HERE/configs)}; shift
OPENPI=${OPENPI:-$HERE/../openpi}
DATA=${DATA:-$HERE/../data}
[ -f "$HERE/configs/$RUN.yaml" ] || { echo "no config $HERE/configs/$RUN.yaml" >&2; exit 2; }
[ -d "$OPENPI/scripts" ] || { echo "no openpi checkout at $OPENPI (set OPENPI=...)" >&2; exit 2; }
[ -d "$DATA/mirror/abc_130k/part_00" ] || { echo "no data at $DATA (run scripts/fetch.py first, or set DATA=...)" >&2; exit 2; }
[ -d "$OPENPI/cdg_run/assets/abc_yam" ] || cp -r "$HERE/cdg_run" "$OPENPI/cdg_run"
cd "$OPENPI"
export OPENPI_CONFIG_DIR=$HERE/configs
export OPENPI_DATA_MIRROR_ROOT=$DATA/mirror
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
# The configs name the base parameters as ../data/checkpoints/...: relative to this checkout.
[ -d ../data/checkpoints/abc_yam_v3/99999/params ] || ln -sfn "$DATA" ../data
if [ -n "${WANDB_API_KEY:-}" ]; then WANDB=(--wandb-enabled --project-name=cdg-finetune); else WANDB=(--no-wandb-enabled); fi
RESTART=--overwrite
for a in "$@"; do [ "$a" = "--resume" ] && RESTART=; done
echo "$(date '+%F %H:%M') $RUN on $(uv run python -c 'import jax; print(len(jax.devices()), "devices")' 2>/dev/null) $*"
exec uv run scripts/train.py "$RUN" --exp-name="$RUN" --checkpoint-base-dir="$OPENPI/checkpoints" "${WANDB[@]}" $RESTART "$@"
