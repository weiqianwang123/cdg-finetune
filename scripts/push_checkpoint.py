"""Upload a run's checkpoint step (its ``params/`` alone) to a Hugging Face
model repo, under ``<run>/<step>/params/``, the layout our evaluation reads.

    HF_TOKEN=... python scripts/push_checkpoint.py --run abc_yam_v3_ft_pour_gt_diverse100 --step 10000
    # --repo qweiw/cdg-checkpoints (default; needs a write token from us) or a repo of your own
    # --checkpoints ../openpi/checkpoints (default: where the fork writes them)
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="the run's name (its --exp-name)")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--repo", default="qweiw/cdg-checkpoints")
    parser.add_argument("--checkpoints", type=Path, default=Path("../openpi/checkpoints"))
    args = parser.parse_args()
    step = args.checkpoints / args.run / args.run / str(args.step) / "params"
    if not step.is_dir():
        raise SystemExit(f"no parameters at {step}")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(step),
        path_in_repo=f"{args.run}/{args.step}/params",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"{args.run} step {args.step} (params)",
    )
    print(f"uploaded to https://huggingface.co/{args.repo}/tree/main/{args.run}/{args.step}")


if __name__ == "__main__":
    main()
