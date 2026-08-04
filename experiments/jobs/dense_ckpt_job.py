"""Replay the first rounds of a campaign FL run with dense checkpointing.

`run_federated_training` is fully seeded, so re-running a campaign seed for the
first R rounds reproduces the same trajectory; the only difference is that we
save a checkpoint every 10 rounds instead of every 100. This gives the early
part of the throughput curve, where the policy actually moves, without
retraining the whole campaign.

Usage:
  py -3 experiments/jobs/dense_ckpt_job.py --save-dir <dir> --method fedavg --seed-idx 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_SEED = 20260701
SEED_OFFSET = {"fedavg": 200, "fedprox": 400}
MU = {"fedavg": 0.0, "fedprox": 0.01}
LOCAL_STEPS = 10


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--method", choices=("fedavg", "fedprox"), required=True)
    parser.add_argument("--seed-idx", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    from main_v3 import run_federated_training

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    run_federated_training(
        # Epsilon annealing is derived from communication_rounds, so the campaign's
        # 1000-round horizon has to stay; the early-check hook stops the loop once
        # the dense part of the curve has been captured.
        communication_rounds=1000,
        early_check_at=args.rounds,
        early_check_callback=lambda _history: {"status": "fail", "reason": "dense-ckpt harvest"},
        local_train_steps=LOCAL_STEPS,
        fedprox_mu=MU[args.method],
        persistent_replay=True,
        replay_memory_size=10000,
        fixed_topology=False,
        save_dir=str(save_dir),
        run_name=f"{args.method}_dense_seed{args.seed_idx:02d}",
        seed=BASE_SEED + args.seed_idx + SEED_OFFSET[args.method],
        checkpoint_every=args.checkpoint_every,
        rho=1.0,
        inference_episodes=0,
    )
    print(f"done {args.method} seed_idx={args.seed_idx}")


if __name__ == "__main__":
    main()
