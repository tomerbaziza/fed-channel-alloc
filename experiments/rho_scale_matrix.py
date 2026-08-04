"""Score every trained policy on both reward scales through one shared harness.

Motivation
----------
The main comparison trained FedAvg/FedProx with rho=1.0 and the centralized
baselines with rho=0.7. Since the CARLTON reward is

    r = rho * r_p + (1 - rho) * r_sw

and r_sw is itself an average of the neighbours' stored (rho * r_p), a symmetric
converged game yields r = rho * (2 - rho) * r_p. That is 0.91 at rho=0.7 and
1.00 at rho=1.0, so a rho=1.0 run reads ~9.9% higher than a rho=0.7 run for the
*same* policy. Raw numbers across the two settings are therefore not comparable.

This script leaves every training run untouched. It reloads the final weights of
each run and replays them, frozen, through `run_inference_episodes` on identical
evaluation scenarios at both rho values. Every method goes through the exact same
evaluation code path, so whatever difference survives is policy quality.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STUDY_DIR = REPO_ROOT / "experiments" / "results" / "main_comparison_rho1_multiseed_20260701"
OUT_PATH = STUDY_DIR / "rho_scale_matrix.json"
RHOS = (0.7, 1.0)

# Same filter the main aggregation applies: PyTorch seeds that actually converged.
PYTORCH_MIN_TRAIN_REWARD = 70.0

TRAINED_RHO = {"pytorch": 0.7, "tf": 0.7, "fedavg": 1.0, "fedprox": 1.0}


def _federated_weights(seed_dir):
    cands = sorted(seed_dir.glob("global_weights_round_*.pkl"))
    return cands[-1] if cands else None


def _pytorch_weights(seed_dir):
    cands = sorted(
        seed_dir.glob("Train_weights_*/trained_weights_step_*.pkl"),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def _pytorch_is_good(seed_dir):
    jc = seed_dir / "job_complete.json"
    if not jc.is_file():
        return False
    try:
        train = json.loads(jc.read_text(encoding="utf-8")).get("last_20_train_reward")
    except json.JSONDecodeError:
        return False
    return train is not None and float(train) >= PYTORCH_MIN_TRAIN_REWARD


def discover(method, n_seeds):
    root = STUDY_DIR / method
    if not root.is_dir():
        return []
    found = []
    for seed_dir in sorted(root.glob("seed_*")):
        if method == "pytorch":
            if not _pytorch_is_good(seed_dir):
                continue
            weights = _pytorch_weights(seed_dir)
        else:
            weights = _federated_weights(seed_dir)
        if weights is not None:
            found.append((seed_dir.name, weights))
    return found[:n_seeds] if n_seeds else found


def _task(job):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from main_v3 import run_inference_episodes

    with open(job["weights"], "rb") as fh:
        weights = pickle.load(fh)
    out = run_inference_episodes(
        global_weights=weights,
        n_episodes=job["episodes"],
        fixed_topology=False,
        rho=job["rho"],
        seed=job["eval_seed"],
    )
    return {
        "method": job["method"],
        "seed_dir": job["seed_dir"],
        "rho": job["rho"],
        "reward": float(np.mean(out["inference_rewards"])),
        "channel_changes": float(np.mean(out["inference_channel_changes"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["pytorch", "fedavg", "fedprox"])
    parser.add_argument("--seeds", type=int, default=0, help="0 = all available")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-seed", type=int, default=987654)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    jobs = []
    for method in args.methods:
        runs = discover(method, args.seeds)
        print(f"{method}: {len(runs)} runs with usable weights", flush=True)
        for seed_name, weights in runs:
            for rho in RHOS:
                jobs.append(
                    {
                        "method": method,
                        "seed_dir": seed_name,
                        "weights": str(weights),
                        "rho": rho,
                        "episodes": args.episodes,
                        # Identical evaluation scenarios across every cell.
                        "eval_seed": args.eval_seed,
                    }
                )

    print(f"Dispatching {len(jobs)} evaluations on {args.workers} workers", flush=True)
    rewards = defaultdict(lambda: defaultdict(list))
    changes = defaultdict(lambda: defaultdict(list))
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            rewards[res["method"]][str(res["rho"])].append(res["reward"])
            changes[res["method"]][str(res["rho"])].append(res["channel_changes"])
            done += 1
            print(
                f"[{done}/{len(jobs)}] {res['method']} {res['seed_dir']} "
                f"rho={res['rho']}: {res['reward']:.2f}",
                flush=True,
            )

    summary = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "episodes_per_eval": args.episodes,
        "eval_seed": args.eval_seed,
        "note": (
            "All methods scored through main_v3.run_inference_episodes on identical "
            "scenarios. Training runs were not modified; FedAvg/FedProx remain rho=1.0."
        ),
        "methods": {},
    }
    for method in rewards:
        entry = {"trained_rho": TRAINED_RHO.get(method)}
        for rho in RHOS:
            vals = rewards[method][str(rho)]
            cc = changes[method][str(rho)]
            entry[str(rho)] = {
                "reward_mean": float(np.mean(vals)),
                "reward_std": float(np.std(vals)),
                "cc_mean": float(np.mean(cc)),
                "n_seeds": len(vals),
                "per_seed_reward": vals,
            }
        entry["ratio_rho1_over_rho07"] = entry["1.0"]["reward_mean"] / entry["0.7"]["reward_mean"]
        summary["methods"][method] = entry

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Same policy, both reward scales ===", flush=True)
    for method, entry in summary["methods"].items():
        print(
            f"{method:9s} (trained rho={entry['trained_rho']}): "
            f"rho=0.7 -> {entry['0.7']['reward_mean']:6.2f} +/- {entry['0.7']['reward_std']:.2f} | "
            f"rho=1.0 -> {entry['1.0']['reward_mean']:6.2f} +/- {entry['1.0']['reward_std']:.2f} | "
            f"ratio {entry['ratio_rho1_over_rho07']:.4f}",
            flush=True,
        )
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
