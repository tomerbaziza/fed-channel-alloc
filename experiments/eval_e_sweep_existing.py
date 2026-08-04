"""Score existing FedProx E-sweep final checkpoints on CQ / Shannon rate.

Understands two layouts:
  * ``E{e}/seed_XX/`` (produced by ``e_sweep_job.py``)
  * ``fedprox_E{e}_mu0p01/`` with weights at the sweep root
    (produced by ``run_parallel_campaign.py``)

Usage:
  py -3 experiments/eval_e_sweep_existing.py \\
      --sweep-dir experiments/results/campaign_20260630/sweeps \\
      --campaign-dir experiments/results/main_comparison_private_buffers_20260801 \\
      --games 24 --out docs/figs/campaign/e_sweep.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.eval_e_sweep import CAMPAIGN_E, evaluate_arm, plot


def discover_arms(sweep_dir: Path, campaign_dir: Path, n_seeds: int):
    arms = {}
    # Layout A: E{e}/seed_*
    for e_dir in sorted(sweep_dir.glob("E*")):
        if e_dir.is_dir() and e_dir.name[1:].isdigit():
            seeds = sorted(e_dir.glob("seed_*"))[:n_seeds]
            if seeds:
                arms[int(e_dir.name[1:])] = seeds
    # Layout B: fedprox_E{e}_mu0p01/ (weights at root — treat as one "seed")
    for d in sorted(sweep_dir.glob("fedprox_E*_mu0p01")):
        e = int(d.name.split("_")[1][1:])
        ckpts = list(d.glob("global_weights_round_*.pkl"))
        if not ckpts:
            # nested Train / run folders
            ckpts = list(d.rglob("global_weights_round_*.pkl"))
        if ckpts:
            arms[e] = [d]  # evaluate_arm looks for seed_dirs with glob pattern
    # Prefer campaign FedProx for E=10 (multi-seed)
    camp = sorted((campaign_dir / "fedprox").glob("seed_*"))[:n_seeds]
    if camp:
        arms[CAMPAIGN_E] = camp
    return arms


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=880001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).resolve()
    campaign_dir = Path(args.campaign_dir).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    arms = discover_arms(sweep_dir, campaign_dir, args.seeds)
    if not arms:
        raise SystemExit(f"no E arms found under {sweep_dir}")
    print("arms:", {e: len(v) for e, v in sorted(arms.items())})

    results = {}
    for e in sorted(arms):
        print(f"=== E={e} ===", flush=True)
        results[str(e)] = evaluate_arm(arms[e], args.games, args.base_seed)

    (out.parent / "e_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results, out, chosen_e=CAMPAIGN_E)

    print(f"\n{'E':>4} {'reward':>16} {'composite':>16} {'Mbps':>16}")
    for e in sorted(int(k) for k in results):
        r = results[str(e)]
        print(
            f"{e:>4} "
            f"{r['reward']['mean']:>8.2f}+-{r['reward']['std']:<6.2f} "
            f"{r['composite']['mean']:>8.3f}+-{r['composite']['std']:<6.3f} "
            f"{r['rate_mbps']['mean']:>8.2f}+-{r['rate_mbps']['std']:<6.2f}"
        )


if __name__ == "__main__":
    main()
