"""Re-evaluate trained FL weights under both rho values to isolate reward scaling.

The main comparison trained FedAvg/FedProx with rho=1.0 and the centralized
baselines with rho=0.7 (the paper setting). Because the total reward is
r = rho * r_p + (1 - rho) * r_sw, the two settings are different objectives and
their accumulated rewards are not directly comparable. This script freezes the
final global weights of the FL runs and scores the *same* policy under both
rho values, so any remaining difference is scale rather than policy quality.
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
RHOS = (1.0, 0.7)
WEIGHTS_NAME = "global_weights_round_1000.pkl"


def _seed_dirs(method, n_seeds):
    root = STUDY_DIR / method
    dirs = sorted(d for d in root.glob("seed_*") if (d / WEIGHTS_NAME).is_file())
    return dirs[:n_seeds]


def _task(job):
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
    parser.add_argument("--methods", nargs="+", default=["fedavg", "fedprox"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=987654)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", default=str(STUDY_DIR / "rho_fairness_check.json"))
    args = parser.parse_args()

    jobs = []
    for method in args.methods:
        for run_dir in _seed_dirs(method, args.seeds):
            for rho in RHOS:
                jobs.append(
                    {
                        "method": method,
                        "seed_dir": run_dir.name,
                        "weights": str(run_dir / WEIGHTS_NAME),
                        "rho": rho,
                        "episodes": args.episodes,
                        # Same evaluation scenarios for both rho values.
                        "eval_seed": args.eval_seed,
                    }
                )

    print(f"Dispatching {len(jobs)} evaluations on {args.workers} workers", flush=True)
    collected = defaultdict(lambda: defaultdict(list))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_task, job): job for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            collected[res["method"]][str(res["rho"])].append(res["reward"])
            print(
                f"{res['method']} {res['seed_dir']} rho={res['rho']}: {res['reward']:.2f}",
                flush=True,
            )

    results = {}
    for method, per_rho in collected.items():
        results[method] = {
            rho: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
                "per_seed": vals,
            }
            for rho, vals in per_rho.items()
        }
        r1 = results[method]["1.0"]["mean"]
        r07 = results[method]["0.7"]["mean"]
        results[method]["ratio_rho1_over_rho07"] = r1 / r07
        print(f"\n{method}: rho=1.0 -> {r1:.2f} | rho=0.7 -> {r07:.2f} | ratio {r1 / r07:.4f}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
