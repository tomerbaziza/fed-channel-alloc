"""Plot FedProx local-steps (E) sensitivity from existing sweep histories + eval.

Combines:
  * training-reward last-20 from ``campaign_20260630`` E-sweep histories, and
  * optional ``e_sweep.json`` from ``eval_e_sweep.py`` (composite / Mbps),

into a single paper figure with up to three panels. When only histories are
available, produces a one-panel reward-vs-E bar/line plot that highlights E=10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history_rewards(sweep_root: Path):
    """Return {E: last20_mean} from fedprox_E*_mu0p01/history.json."""
    out = {}
    for d in sorted(sweep_root.glob("fedprox_E*_mu0p01")):
        hist = d / "history.json"
        if not hist.exists():
            continue
        h = json.loads(hist.read_text(encoding="utf-8"))
        e = int(h.get("config", {}).get("local_train_steps", d.name.split("_")[1][1:]))
        r = h["round_rewards"]
        out[e] = float(np.mean(r[-20:]))
    return out


def plot_from_eval(results: dict, out_path: Path, chosen_e: int = 10):
    es = sorted(int(e) for e in results)
    panels = [
        ("reward", "Inference reward", "#4C78A8"),
        ("composite", r"$\mathbb{E}[(\mathrm{CQ}+\min_{\mathrm{CQ}})/2]$", "#54A24B"),
        ("rate_mbps", "Throughput [Mbps]", "#F58518"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, (key, ylabel, color) in zip(axes, panels):
        means = np.array([results[str(e)][key]["mean"] for e in es])
        stds = np.array([results[str(e)][key]["std"] for e in es])
        ax.errorbar(es, means, yerr=stds, marker="o", color=color,
                    capsize=3, linewidth=1.8, markersize=6)
        best = es[int(np.nanargmax(means))]
        ax.axvline(chosen_e, color="0.55", linestyle="--", linewidth=1.1,
                   label=f"$E={chosen_e}$ (chosen)")
        ax.scatter([best], [means[es.index(best)]], marker="*", s=190,
                   color=color, edgecolor="black", zorder=5, linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xticks(es)
        ax.set_xticklabels([str(e) for e in es])
        ax.set_xlabel("Local steps $E$")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print("wrote", out_path)


def plot_from_histories(rewards: dict, out_path: Path, chosen_e: int = 10):
    es = sorted(rewards)
    vals = [rewards[e] for e in es]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(es, vals, marker="o", color="#4C78A8", linewidth=2.0, markersize=7)
    best = es[int(np.argmax(vals))]
    ax.axvline(chosen_e, color="0.55", linestyle="--", linewidth=1.1,
               label=f"$E={chosen_e}$ (chosen)")
    ax.scatter([best], [rewards[best]], marker="*", s=220, color="#F58518",
               edgecolor="black", zorder=5, linewidth=0.7, label=f"best ($E={best}$)")
    ax.set_xscale("log")
    ax.set_xticks(es)
    ax.set_xticklabels([str(e) for e in es])
    ax.set_xlabel("Local steps $E$")
    ax.set_ylabel("Train reward (last-20 mean)")
    ax.set_title(r"FedProx sensitivity to local steps $E$ ($\mu=0.01$)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print("wrote", out_path)
    return {"by_e": {str(e): rewards[e] for e in es}, "best_e": best, "chosen_e": chosen_e}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-sweep-dir", default=None,
                        help="campaign_*/sweeps with fedprox_E*_mu0p01")
    parser.add_argument("--eval-json", default=None,
                        help="e_sweep.json from eval_e_sweep.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chosen-e", type=int, default=10)
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.eval_json and Path(args.eval_json).exists():
        results = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
        plot_from_eval(results, out, chosen_e=args.chosen_e)
    elif args.history_sweep_dir:
        rewards = load_history_rewards(Path(args.history_sweep_dir).resolve())
        summary = plot_from_histories(rewards, out, chosen_e=args.chosen_e)
        out.with_name("e_sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    else:
        raise SystemExit("provide --eval-json or --history-sweep-dir")


if __name__ == "__main__":
    main()
