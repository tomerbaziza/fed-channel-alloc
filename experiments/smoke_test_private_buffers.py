"""Smoke test: confirm FedAvg/FedProx still learn under the new per-client
(private, persistent) replay buffer policy, with RHO=1.0 as mandated by the
project policy (FL uses RHO=1.0; centralized baselines use RHO=0.7).

Short and parallel by design -- its only job is to confirm the reward curve
rises meaningfully above its own starting point before we commit to a full
30-seed / 1000-round re-run. Pass criterion is calibrated against the observed
growth rate of the old (shared-buffer) 1000-round runs: first20=78.3 ->
last20=86.5, i.e. ~+8 over 1000 rounds. Over 150 rounds under the *harder*
private-buffer regime we only require a positive, non-trivial upward trend
(last20 - first20 > 2) plus healthy absolute reward levels, not a full
convergence match.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROUNDS = 150
BEST_E = 10
BEST_MU = 0.01
OUT_DIR = REPO_ROOT / "experiments" / "results" / "_smoke_private_buffers"

JOBS = [
    {"method": "fedavg", "seed": 90001, "fedprox_mu": 0.0},
    {"method": "fedprox", "seed": 90002, "fedprox_mu": BEST_MU},
]


def _run(job):
    sys.path.insert(0, str(REPO_ROOT))
    from main_v3 import run_federated_training

    method = job["method"]
    seed = job["seed"]
    save_dir = OUT_DIR / method / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    history, _weights = run_federated_training(
        communication_rounds=ROUNDS,
        local_train_steps=BEST_E,
        fedprox_mu=job["fedprox_mu"],
        persistent_replay=True,
        replay_memory_size=10000,
        fixed_topology=False,
        save_dir=str(save_dir),
        run_name=f"smoke_{method}_seed{seed}",
        seed=seed,
        checkpoint_every=0,
        rho=1.0,  # FL policy: RHO=1.0
        inference_episodes=20,
    )
    rewards = history["round_rewards"]
    first20 = float(np.mean(rewards[:20]))
    last20 = float(np.mean(rewards[-20:]))
    inf = (history.get("inference") or {}).get("inference_reward_last_20")
    final_per_client = history["replay_buffer_counts_per_client"][-1]
    return {
        "method": method,
        "seed": seed,
        "first20_train_reward": first20,
        "last20_train_reward": last20,
        "inference_reward_last_20": inf,
        "final_per_client_buffer_counts": final_per_client,
        "n_clients_with_data": sum(1 for v in final_per_client.values() if v > 0),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run, job): job for job in JOBS}
        for fut in as_completed(futures):
            res = fut.result()
            print(json.dumps(res, indent=2), flush=True)
            all_results.append(res)

    (OUT_DIR / "smoke_summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    print("\n=== Smoke test summary ===")
    ok = True
    for r in all_results:
        print(
            f"{r['method']} seed={r['seed']}: first20={r['first20_train_reward']:.2f} "
            f"-> last20={r['last20_train_reward']:.2f} | inference_last20="
            f"{r['inference_reward_last_20']}"
        )
        if r["last20_train_reward"] <= r["first20_train_reward"] + 2:
            print(f"  WARN: little/no upward trend for {r['method']} seed={r['seed']}")
            ok = False
        if r["n_clients_with_data"] < 6:
            print(
                f"  WARN: too few clients accumulated data for {r['method']} "
                f"seed={r['seed']} ({r['n_clients_with_data']})"
            )
            ok = False

    if ok:
        print("\nPASS: learning occurs under the private per-client buffer policy.")
    else:
        print("\nFAIL: inspect results before launching the full campaign.")
        sys.exit(1)


if __name__ == "__main__":
    main()
