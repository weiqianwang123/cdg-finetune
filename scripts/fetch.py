"""Fetch what the runs need from Hugging Face: the datasets' mirror
(qweiw/cdg-abc-data, folder mirror/: about 120 GB) and the base checkpoint's
parameters (qweiw/cdg-checkpoints, abc_yam_v3/99999/params/: 11.6 GB).

    python scripts/fetch.py --root ../data          # everything (resumable)
    python scripts/fetch.py --root ../data --check  # what is still missing
    python scripts/fetch.py --root ../data --only pour   # one domain's datasets, with the mix

Lays out ``<root>/mirror/abc_sim_224/<name>/``, ``<root>/mirror/abc_130k/part_0k/``
and ``<root>/checkpoints/abc_yam_v3/99999/params/``. Needs ``huggingface_hub``
(``uv run --with huggingface_hub python scripts/fetch.py ...``). No token is
needed: both repos are public.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DATA_REPO = "qweiw/cdg-abc-data"
CHECKPOINT_REPO = "qweiw/cdg-checkpoints"
BASE = "abc_yam_v3/99999"
MIX = [  # the base's training mix, half of every batch
    "ball_tray_balancing", "conveyor_pick", "count_into_opaque_box", "grab_clutter",
    "lego_blocks_sorting", "multi_drawer_search", "nuts_bolts_sorting", "put_relative",
    "sim_inhand_transfer_the_item_to_other_side", "sim_set_up_chess_pieces_on_the_board",
    "sim_spell_abc", "sim_spell_agi", "sim_spell_cat", "sim_spell_fish", "sim_spell_xdof",
    "sim_spell_yam", "sim_throw_plastic_bottles_in_bin",
]  # fmt: skip
PARTS = [f"part_0{k}" for k in range(8)]
SETS = {  # the fine-tune datasets, by domain
    "pour": ["pour_gt_diverse100", "pour_gt_diverse_style100"],
    "hang_mug": ["hang_mug_mug6_gt_diverse100", "hang_mug_mug6_gt_diverse_style100"],
    "sweep_only": ["sweep_only_gt_diverse100", "sweep_only_gt_diverse_style100"],
}


def patterns(only: str | None) -> list[str]:
    """The mirror's folders to fetch, as allow patterns."""
    sets = [s for domain, names in SETS.items() if only in (None, domain) for s in names]
    return (
        [f"mirror/abc_sim_224/{name}/*" for name in MIX + sets]
        + [f"mirror/abc_130k/{part}/*" for part in PARTS]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("../data"))
    parser.add_argument("--only", choices=sorted(SETS), help="one domain's datasets")
    parser.add_argument("--check", action="store_true", help="list what is missing")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.check:
        missing = []
        for name in MIX + [s for names in SETS.values() for s in names]:
            if not (root / "mirror" / "abc_sim_224" / name / "meta" / "info.json").is_file():
                missing.append(f"mirror/abc_sim_224/{name}")
        for part in PARTS:
            if not (root / "mirror" / "abc_130k" / part / "meta" / "info.json").is_file():
                missing.append(f"mirror/abc_130k/{part}")
        if not (root / "checkpoints" / BASE / "params").is_dir():
            missing.append(f"checkpoints/{BASE}/params")
        print("\n".join(missing) if missing else "everything is here")
        return
    snapshot_download(
        DATA_REPO,
        repo_type="dataset",
        allow_patterns=patterns(args.only),
        local_dir=root,
        max_workers=8,
    )
    snapshot_download(
        CHECKPOINT_REPO,
        repo_type="model",
        allow_patterns=[f"{BASE}/params/*"],
        local_dir=root / "checkpoints",
        max_workers=8,
    )
    print(f"fetched into {root}")


if __name__ == "__main__":
    main()
