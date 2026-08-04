"""One FedProx run for the local-steps (E) sensitivity study.

Seeds and every other hyperparameter match the main campaign
(`run_main_comparison_multiseed.py`), so the E=10 arm can be read directly from
that campaign instead of being retrained: seed = 20260701 + idx + 400,
rho = 1.0, mu = 0.01, persistent private buffers, 1000 rounds.

Usage:
  py -3 experiments/jobs/e_sweep_job.py --save-dir <dir> --seed-idx 0 --local-steps 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_SEED = 20260701
FEDPROX_SEED_OFFSET = 400
MU = 0.01
ROUNDS = 1000
INFERENCE_EPISODES = 100


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed-idx", type=int, required=True)
    parser.add_argument("--local-steps", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--inference-episodes", type=int, default=INFERENCE_EPISODES)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    from main_v3 import run_federated_training

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "job_complete.json").exists():
        print("already complete, skipping")
        return

    seed = BASE_SEED + args.seed_idx + FEDPROX_SEED_OFFSET
    e = args.local_steps

    history, _weights = run_federated_training(
        communication_rounds=args.rounds,
        local_train_steps=e,
        fedprox_mu=MU,
        persistent_replay=True,
        replay_memory_size=10000,
        fixed_topology=False,
        save_dir=str(save_dir),
        run_name=f"fedprox_E{e}_mu{MU}_seed{args.seed_idx:02d}",
        seed=seed,
        checkpoint_every=200,
        rho=1.0,
        inference_episodes=args.inference_episodes,
    )

    inference = history.get("inference") or {}
    (save_dir / "job_complete.json").write_text(
        json.dumps(
            {
                "local_train_steps": e,
                "seed_idx": args.seed_idx,
                "seed": seed,
                "mu": MU,
                "last_20_train_reward": float(np.mean(history["round_rewards"][-20:])),
                "inference_reward_mean": inference.get("inference_reward_mean"),
                "inference_reward_last_20": inference.get("inference_reward_last_20"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done E={e} seed_idx={args.seed_idx}")


if __name__ == "__main__":
    main()
