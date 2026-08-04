"""Aggregate Shannon-throughput JSON shards into the paper Fig. 4 curve.

Reads every ``{method}_seed_XX.json`` produced by ``eval_throughput_curve.py``
and writes a single mean+/-std plot of network-average rate [Mbps] versus
training round for FedAvg and FedProx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"fedavg": "#F58518", "fedprox": "#54A24B"}
LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx"}


def load_method(shards_dir: Path, method: str):
    files = sorted(shards_dir.glob(f"{method}_seed_*.json"))
    if not files:
        raise FileNotFoundError(f"no shards for {method} under {shards_dir}")
    per_seed = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rounds = {int(r): float(np.mean(v["mean"])) for r, v in d["rounds"].items()}
        per_seed.append(rounds)
    all_rounds = sorted(set().union(*[set(s) for s in per_seed]))
    means, stds = [], []
    for r in all_rounds:
        vals = [s[r] for s in per_seed if r in s]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    return all_rounds, np.asarray(means), np.asarray(stds), len(per_seed)


def plot(shards_dir: Path, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    summary = {}
    for method in ("fedavg", "fedprox"):
        rounds, means, stds, n = load_method(shards_dir, method)
        ax.plot(rounds, means, color=COLORS[method], linewidth=2.0,
                label=f"{LABELS[method]} (n={n})")
        ax.fill_between(rounds, means - stds, means + stds,
                         color=COLORS[method], alpha=0.22, linewidth=0)
        summary[method] = {
            "rounds": rounds,
            "mean_mbps": means.tolist(),
            "std_mbps": stds.tolist(),
            "n_seeds": n,
            "final_mean": float(means[-1]),
            "final_std": float(stds[-1]),
        }
    ax.set_xlabel("Training round")
    ax.set_ylabel(r"Throughput $B\log_2(1+\mathrm{SINR})$ [Mbps]")
    ax.set_title("Shannon rate of the learned allocation")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    (out_path.parent / "throughput_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("wrote", out_path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    plot(Path(args.shards_dir).resolve(), Path(args.out).resolve())


if __name__ == "__main__":
    main()
