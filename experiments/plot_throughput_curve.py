"""Aggregate Shannon-throughput shards into paper Fig. 4 (option D).

Two panels:
  (a) absolute Mbps with light smoothing
  (b) learning curve normalized by the final-round rate: R(t) / R(T)
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
    return np.asarray(all_rounds), np.asarray(means), np.asarray(stds), len(per_seed)


def smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Centered moving average; window=1 leaves the series unchanged."""
    if window <= 1 or len(y) < 2:
        return y.copy()
    w = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w < 3:
        return y.copy()
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(yp, kernel, mode="valid")


def plot_dual(shards_dir: Path, out_path: Path, smooth_window: int = 3):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    summary = {}

    for method in ("fedavg", "fedprox"):
        rounds, means, stds, n = load_method(shards_dir, method)
        means_s = smooth(means, smooth_window)
        stds_s = smooth(stds, smooth_window)
        final = float(means[-1])
        norm = means / final
        norm_s = means_s / final
        norm_std = stds / final

        color, label = COLORS[method], f"{LABELS[method]} (n={n})"

        axes[0].plot(rounds, means_s, color=color, linewidth=2.0, label=label)
        axes[0].fill_between(
            rounds, means_s - stds_s, means_s + stds_s,
            color=color, alpha=0.20, linewidth=0,
        )
        axes[0].plot(rounds, means, color=color, linewidth=0.8, alpha=0.35)

        axes[1].plot(rounds, norm_s, color=color, linewidth=2.0, label=label)
        axes[1].fill_between(
            rounds, np.clip(norm_s - norm_std, 0, None), norm_s + norm_std,
            color=color, alpha=0.20, linewidth=0,
        )
        axes[1].axhline(1.0, color="0.55", linestyle="--", linewidth=1.0)

        summary[method] = {
            "rounds": rounds.tolist(),
            "mean_mbps": means.tolist(),
            "mean_mbps_smooth": means_s.tolist(),
            "std_mbps": stds.tolist(),
            "normalized": norm.tolist(),
            "normalized_smooth": norm_s.tolist(),
            "n_seeds": n,
            "final_mean": final,
            "final_std": float(stds[-1]),
        }

    axes[0].set_xlabel("Training round")
    axes[0].set_ylabel(r"$B\log_2(1+\mathrm{SINR})$ [Mbps]")
    axes[0].set_title("(a) Absolute throughput")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)

    axes[1].set_xlabel("Training round")
    axes[1].set_ylabel(r"$R(t)\,/\,R(T)$")
    axes[1].set_title("(b) Normalized learning curve")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[1].set_ylim(0.55, 1.15)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    (out_path.parent / "throughput_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("wrote", out_path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smooth-window", type=int, default=3)
    args = parser.parse_args(argv)
    plot_dual(
        Path(args.shards_dir).resolve(),
        Path(args.out).resolve(),
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()
