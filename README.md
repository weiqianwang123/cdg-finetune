# cdg-finetune: π0.5 fine-tunes on CDG's ABC-Sim program data, on GPUs

Six fine-tunes of the same base policy (π0.5 post-trained on the dual-YAM arms in
ABC-Sim), one per dataset, with one fixed recipe. Everything needed is public:

| what | where |
|---|---|
| training code | the openpi fork [weiqianwang123/openpi](https://github.com/weiqianwang123/openpi) at commit `df653adb3e1bf957ed5897fb7b62a6209f872be2` (pinned below) |
| base checkpoint | Hugging Face model repo `qweiw/cdg-checkpoints`, folder `abc_yam_v3/99999/params/` (the parameters alone, 11.6 GB; `train_state/` is not needed) |
| datasets | Hugging Face dataset repo `qweiw/cdg-abc-data`, folder `mirror/` (LeRobot v3.0 datasets: the six fine-tune sets and the training mix, about 120 GB) |
| recipes | `configs/*.yaml` in this repo, one per run; `cdg_run/` (norm statistics and the datasets' non-idle frame ranges) |

This README is written to be followed end to end by a person or by a coding agent.
Every command is given in full. Nothing here needs Google Cloud or TPUs.

## 0. What a run is

Each run fine-tunes the base for **10,000 steps at batch 128**, mixing its own dataset
(50% of every batch) with the base's training mix (50%: 17 ABC-Sim tasks and 8 parts of
ABC's 130k human episodes). Learning rate: cosine from 2.5e-5 after 1,000 warm-up steps,
with a 50,000-step decay horizon (the run stops at 10,000, where the rate is still 2.3e-5:
this matches the runs we compare against, which were configured for 50k and compared at
10k). Absolute joint actions, action horizon 30, images at 224. A checkpoint is saved every
1,000 steps; **the one at step 10000 is the result**.

| run (`--exp-name`, also the config name) | dataset (`mirror/abc_sim_224/<name>`) | domain | data |
|---|---|---|---|
| `abc_yam_v3_ft_pour_gt_diverse100` | `pour_gt_diverse100` | pouring | program asked for diverse data |
| `abc_yam_v3_ft_pour_gt_diverse_style100` | `pour_gt_diverse_style100` | pouring | program asked for diverse data, style-aligned |
| `abc_yam_v3_ft_hangmug_gt_diverse100` | `hang_mug_mug6_gt_diverse100` | hang mug | program asked for diverse data |
| `abc_yam_v3_ft_hangmug_gt_diverse_style100` | `hang_mug_mug6_gt_diverse_style100` | hang mug | program asked for diverse data, style-aligned |
| `abc_yam_v3_ft_sweeponly_gt_diverse100` | `sweep_only_gt_diverse100` | sweep only | program asked for diverse data |
| `abc_yam_v3_ft_sweeponly_gt_diverse_style100` | `sweep_only_gt_diverse_style100` | sweep only | program asked for diverse data, style-aligned |

Each dataset is 100 successful episodes of a coding agent's program run in ABC-Sim. The
runs are independent: run them in any order, in parallel if the GPUs allow.

## 1. Hardware: 8-GPU nodes

The model is π0.5 (about 3.4B parameters), fine-tuned in full (no LoRA) at global batch 128.
Per GPU it needs roughly **94 GB / N + 3 GB**, where N is the number of GPUs the model is
sharded over (`--fsdp-devices N`: parameters and optimizer state shard N ways, and so does
the batch; the GPUs beyond N run as further data-parallel groups). Measured anchors: on an
8-device TPU v6e (32 GB each) batch 256 fit with N = 8; on a 4-device TPU v4 (31 GB each)
batch 256 did not fit with N = 4 (35 GB per device needed).

| node | one run on the node | two runs at once (4 GPUs each) | per-GPU memory |
|---|---|---|---|
| 8 × H100 80 GB | `--fsdp-devices 8` (safe) or `--fsdp-devices 4` (two data-parallel groups, faster) | `CUDA_VISIBLE_DEVICES=0,1,2,3 ... --fsdp-devices 4` and `4,5,6,7` | ~15 GB (N = 8), ~27 GB (N = 4) |
| 8 × H200 141 GB | `--fsdp-devices 4` or `--fsdp-devices 2` (four groups) | the same, `--fsdp-devices 2` fits too | ~27 GB (N = 4), ~50 GB (N = 2) |
| 8 × B200 | `--fsdp-devices 2` (four groups) | the same | ~50 GB (N = 2) |

Smaller N is faster per step (less sharding traffic) as long as it fits; when in doubt start
with `--fsdp-devices 8` and, if it runs, try a smaller N on the next run. The six runs are
independent: with 8 GPUs a node can run two at once with 4 GPUs each.

Rules:
- Keep `batch_size: 128` as it is in every config. Do not change it, and do not use LoRA:
  the results are compared against runs with exactly this recipe.
- `--fsdp-devices` must divide the number of GPUs JAX sees (`CUDA_VISIBLE_DEVICES`) and must
  divide 128.
- `scripts/train.sh` exports `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (JAX otherwise preallocates
  less and fails with a false out-of-memory).
- If a run fails with `RESOURCE_EXHAUSTED` in its first step, double `--fsdp-devices` and
  start again; the log's "Used X of Y hbm" line says by how much it missed.
- Throughput to expect: on 8 × H100 about 1–1.5 s per step, so 10,000 steps take 3–5 hours;
  H200/B200 faster. The first step takes several minutes (compilation).
- Blackwell (B200): the pinned `jax[cuda12]==0.5.3` may lack kernels for the newest GPUs. If
  JAX does not see the GPUs or fails to compile, run on H100/H200 instead, or try the newest
  `jax[cuda12]` of the same minor version (`uv pip install -U "jax[cuda12]"` inside the fork's
  environment) and report what was changed.

## 2. Setup (once)

Requirements: Linux, NVIDIA driver with CUDA 12 support, `uv` (https://docs.astral.sh/uv/),
`git`, `git-lfs`, Python 3.11 (uv installs it), about 150 GB of disk for the datasets and
checkpoints, outbound access to github.com and huggingface.co.

Three folders side by side under one working directory (the scripts assume this layout;
`OPENPI=` and `DATA=` override it):

```
work/
  openpi/         the training code (the fork, pinned)
  cdg-finetune/   this repo (configs, scripts, cdg_run)
  data/           the datasets' mirror and the base checkpoint (scripts/fetch.py)
```

```bash
mkdir work && cd work
git clone https://github.com/weiqianwang123/cdg-finetune.git
git clone https://github.com/weiqianwang123/openpi.git
cd openpi
git checkout df653adb3e1bf957ed5897fb7b62a6209f872be2
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ..
```

Check that JAX sees the GPUs:

```bash
cd openpi && uv run python -c "import jax; print(jax.devices())"
```

## 3. Data and the base checkpoint (once, about 130 GB)

```bash
cd work/cdg-finetune
uv run --with huggingface_hub python scripts/fetch.py --root ../data
```

This downloads `qweiw/cdg-abc-data/mirror/` to `work/data/mirror/` (the LeRobot datasets,
laid out as `mirror/abc_sim_224/<name>/...` and `mirror/abc_130k/part_0k/...`) and the base
parameters to `work/data/checkpoints/abc_yam_v3/99999/params/`. It is resumable: run it
again if it stops. `uv run --with huggingface_hub python scripts/fetch.py --root ../data --check`
lists what is still missing.

## 4. Train (one command per run)

```bash
cd work/cdg-finetune
bash scripts/train.sh abc_yam_v3_ft_pour_gt_diverse100 --fsdp-devices 8
```

`scripts/train.sh RUN [openpi train.py options]` copies `cdg_run/` into the fork (the
configs refer to it as `cdg_run/assets` and `cdg_run/keep/...`), sets the environment the
fork reads and runs its `scripts/train.py` from the fork's folder:

- `OPENPI_CONFIG_DIR=work/cdg-finetune/configs` — where the run's yaml is found by name;
- `OPENPI_DATA_MIRROR_ROOT=work/data/mirror` — the datasets' root (a plain directory);
- the base parameters: the yaml's `weight_loader.params_path` is
  `../data/checkpoints/abc_yam_v3/99999/params`, relative to the fork's folder (the layout
  above; with `DATA=` elsewhere the script links `work/data` to it);
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`;
- `--exp-name=RUN --checkpoint-base-dir=work/openpi/checkpoints --overwrite`, and
  `--no-wandb-enabled` unless `WANDB_API_KEY` is set (then `--wandb-enabled
  --project-name=cdg-finetune`).

Any further option is passed to `scripts/train.py` as is (tyro's command line: `--fsdp-devices 8`,
`--num-train-steps 20`, `--resume`...). Checkpoints go to
`work/openpi/checkpoints/<RUN>/<RUN>/<step>/` (every 1,000 steps; `params/` is what matters).
A run interrupted is resumed by adding `--resume` to the same command.

Run the six in sequence or in parallel; nothing is shared between them but the data.

## 5. What to send back

For every run, the checkpoint at step 10000, parameters only:
`openpi/checkpoints/<config>/<RUN>/10000/params/` (about 11.6 GB each), plus the console log
(or the W&B link). Upload them to Hugging Face:

```bash
# with a write token for the model repo (ask us for one, or use a repo of your own and tell us its name)
cd work/cdg-finetune
HF_TOKEN=... uv run --with huggingface_hub python scripts/push_checkpoint.py \
    --run abc_yam_v3_ft_pour_gt_diverse100 --step 10000 --repo qweiw/cdg-checkpoints
```

This puts `params/` under `<RUN>/10000/` in the repo, in the layout our evaluation reads.
The checkpoints at 5000 are welcome too (`--step 5000`) but not required. We run the
evaluation in ABC-Sim on our side.

## 6. Checks before a long run

- `bash scripts/train.sh abc_yam_v3_ft_pour_gt_diverse100 --fsdp-devices 8 --num-train-steps 20 --save-interval 20`
  runs 20 steps end to end (data streaming, the base loaded, one checkpoint written) in a
  few minutes; a failure here is a setup problem, not a training one. Delete
  `work/openpi/checkpoints/abc_yam_v3_ft_pour_gt_diverse100` afterwards (the real run
  overwrites it anyway).
- A healthy run's loss starts around 1.5 and is below 0.05 within the first 2,000 steps.
- The data mix is read by streaming: the first steps may be slow while the shuffle buffer
  (8,000 samples) fills.

## 7. Where things come from

- The base `abc_yam_v3` is π0.5 post-trained by us on ABC-Sim's data (17 tasks) mixed with
  ABC's 130k human episodes; the mix here is a 224-pixel conversion of ABC's public data
  (https://huggingface.co/datasets/abc-sim) in LeRobot v3 format, re-hosted for this purpose.
- The six datasets were generated by coding agents' programs in ABC-Sim (the CDG project);
  each episode has the three cameras at 224 pixels, 14-dimensional states and actions at 30 Hz,
  and `meta/keep_ranges.json` marks the frames without idle motion that the recipe trains on.
- Questions: Qianwei Wang (qweiw@umich.edu).
