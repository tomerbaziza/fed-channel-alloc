"""Sensitivity of \\OURS{} to the number of local steps E between aggregations.

For each E we score the final global model of every seed on a common in-sample
scenario set (N in 2..7) and report the three quantities that matter
operationally: post-training reward, composite channel quality
E[(CQ+min_CQ)/2], and Shannon throughput B*log2(1+SINR).

E=10 is read from the main campaign (identical seeds and hyperparameters); the
other arms come from the dedicated E-sweep study.

Usage:
  py -3 experiments/eval_e_sweep.py --sweep-dir experiments/results/e_sweep_20260804 \\
      --campaign-dir experiments/results/main_comparison_private_buffers_20260801 \\
      --games 30
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BuildingBlocks.Worker import worker
from SimulationEnvironments.Pythonic_Environment import python_env
from Utils.RandomLocationOfNetworks import set_random_location_of_networks

from experiments.eval_paper_fig15 import _metrics_from_history
from experiments.eval_throughput_curve import scenario_rate_mbps

K = 10
MIN_NETS, MAX_NETS = 2, 7
CAMPAIGN_E = 10


def score_episode(weights, n_nets: int, seed: int):
    np.random.seed(int(seed))
    users, centers = set_random_location_of_networks(n_nets)
    env = python_env(
        number_of_nets=n_nets,
        number_of_users_in_each_net=users,
        net_center_location_and_std=centers,
        possible_channels=K,
        add_noise=False,
        training=False,
    )
    _, _, _, game_history = worker(
        address_scen="",
        scenario=env,
        address_algo="",
        training=False,
        epsilon=0.0,
        global_weights=weights,
        local_train_steps=0,
        save_to_global_rb=False,
        verbose=False,
    )
    metrics = _metrics_from_history(game_history, n_nets, K)
    metrics["rate_mbps"] = scenario_rate_mbps(env)
    return metrics


def final_checkpoint(seed_dir: Path):
    cands = sorted(
        seed_dir.glob("global_weights_round_*.pkl"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    return cands[-1] if cands else None


def inference_reward(seed_dir: Path):
    for name in ("job_complete.json", "history.json"):
        f = seed_dir / name
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        val = d.get("inference_reward_mean")
        if val is None:
            val = (d.get("inference") or {}).get("inference_reward_mean")
        if val is not None:
            return float(val)
    return float("nan")


def evaluate_arm(seed_dirs, games, base_seed):
    scenarios = []
    for g in range(games):
        seed = base_seed + g
        rng = np.random.RandomState(seed)
        scenarios.append((seed, int(rng.randint(MIN_NETS, MAX_NETS + 1))))

    per_seed = []
    for sd in seed_dirs:
        ckpt = final_checkpoint(sd)
        if ckpt is None:
            print(f"  !! no checkpoint in {sd}")
            continue
        with ckpt.open("rb") as f:
            weights = pickle.load(f)
        rows = [score_episode(weights, n, s) for s, n in scenarios]
        per_seed.append(
            {
                "seed_dir": str(sd),
                "reward": inference_reward(sd),
                "composite": float(np.mean([r["composite"] for r in rows])),
                "rate_mbps": float(np.mean([r["rate_mbps"] for r in rows])),
                "ws": float(np.mean([r["ws"] for r in rows])),
            }
        )
        print(
            f"  {sd.parent.name}/{sd.name}: reward={per_seed[-1]['reward']:.2f} "
            f"composite={per_seed[-1]['composite']:.3f} "
            f"rate={per_seed[-1]['rate_mbps']:.2f} Mbps",
            flush=True,
        )

    def agg(key):
        vals = [p[key] for p in per_seed if not np.isnan(p[key])]
        return {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "n": len(vals),
        }

    return {k: agg(k) for k in ("reward", "composite", "rate_mbps", "ws")} | {
        "per_seed": per_seed
    }


PANELS = [
    ("composite", r"$\mathbb{E}[(\mathrm{CQ}+\min_{\mathrm{CQ}})/2]$", "#54A24B"),
    ("rate_mbps", "Throughput [Mbps]", "#F58518"),
]


def plot(results, out_path: Path, chosen_e: int = 10):
    es = sorted(int(e) for e in results)
    xpos = np.arange(len(es))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, (key, ylabel, color) in zip(axes, PANELS):
        means = np.array([results[str(e)][key]["mean"] for e in es])
        stds = np.array([results[str(e)][key]["std"] for e in es])
        ax.errorbar(
            xpos, means, yerr=stds, marker="o", color=color,
            capsize=3, linewidth=1.8, markersize=6,
        )
        best_i = int(np.nanargmax(means))
        if chosen_e in es:
            ax.axvline(es.index(chosen_e), color="0.55", linestyle="--", linewidth=1.1,
                       label=f"$E={chosen_e}$ (chosen)")
        ax.scatter([xpos[best_i]], [means[best_i]], marker="*", s=190,
                   color=color, edgecolor="black", zorder=5, linewidth=0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(e) for e in es], rotation=45, ha="right", fontsize=7.5)
        ax.set_xlim(-0.4, len(es) - 0.6)
        ax.set_xlabel("Local steps $E$")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print("wrote", out_path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=880001)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    sweep_dir = Path(args.sweep_dir).resolve()
    campaign_dir = Path(args.campaign_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else sweep_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = {}
    for e_dir in sorted(sweep_dir.glob("E*")):
        if not e_dir.is_dir():
            continue
        arms[int(e_dir.name[1:])] = sorted(e_dir.glob("seed_*"))[: args.seeds]
    arms[CAMPAIGN_E] = sorted((campaign_dir / "fedprox").glob("seed_*"))[: args.seeds]

    results = {}
    for e in sorted(arms):
        print(f"=== E={e} ===", flush=True)
        results[str(e)] = evaluate_arm(arms[e], args.games, args.base_seed)

    (out_dir / "e_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot(results, out_dir / "e_sweep.png", chosen_e=CAMPAIGN_E)

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
