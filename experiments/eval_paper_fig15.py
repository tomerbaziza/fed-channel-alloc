"""Paper-style algorithm comparison (Fig. 15 / Fig. 13).

Evaluates trained policies and heuristic baselines on the CARLTON protocol:
  30 games x scenarios with N in {2,...,15} = 420 games,
then plots E[(CQ + min_CQ)/2] as:
  - grouped bars: in-sample (N<=7) vs out-of-sample (N>7)   [Fig. 15]
  - curves vs number of networks                           [Fig. 13]

Methods:
  - FedAvg / FedProx / PyTorch CTDE  (frozen DeepMellow weights)
  - RA   (random initial channel, never switch)
  - JAR  (+/-1 channel iff CQ gain >= 0.05)

Usage:
  py -3 experiments/eval_paper_fig15.py \\
      --study-dir experiments/results/main_comparison_rho1_multiseed_20260701 \\
      --games-per-n 30 --max-workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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
from Utils.ScenarioExamination import get_game_performamce
from Utils.save_to_df_csv import wrrape_game_history_do_df

K = 10
N_RANGE = list(range(2, 16))  # paper: 2..15
IN_SAMPLE_MAX = 7
JAR_DELTA = 0.05

METHOD_LABELS = {
    "pytorch": "PyTorch CTDE",
    "carlton_paper": "CARLTON (paper replica)",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "ra": "RA",
    "jar": "JAR",
}

METHOD_COLORS = {
    "pytorch": "#4C78A8",
    "carlton_paper": "#4C78A8",
    "fedavg": "#F58518",
    "fedprox": "#54A24B",
    "ra": "#B279A2",
    "jar": "#E45756",
}


# ---------------------------------------------------------------------------
# Weight discovery
# ---------------------------------------------------------------------------

def _latest_pytorch_weights(seed_dir: Path):
    # Checkpoint tags are unpadded replay-buffer counters, so lexicographic
    # order puts "step_9980" after "step_78620". Order by write time, which is
    # what `creat_player` uses when it reloads weights during training.
    cands = sorted(
        seed_dir.glob("Train_weights_*/trained_weights_step_*.pkl"),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def _latest_federated_weights(seed_dir: Path):
    cands = sorted(seed_dir.glob("global_weights_round_*.pkl"))
    return cands[-1] if cands else None


def discover_weight_methods(study_dir: Path, max_seeds: int | None = None):
    """Return {method: [Path, ...]} of final weight files, one per seed."""
    found = {}
    for method, finder in (
        ("pytorch", _latest_pytorch_weights),
        ("fedavg", _latest_federated_weights),
        ("fedprox", _latest_federated_weights),
    ):
        roots = sorted((study_dir / method).glob("seed_*"))
        if max_seeds is not None:
            roots = roots[:max_seeds]
        paths = []
        for sd in roots:
            p = finder(sd)
            if p is not None:
                paths.append(p)
        if paths:
            found[method] = paths
    return found


def _load_weights(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Heuristic policies
# ---------------------------------------------------------------------------

def jar_action(qv, current_ch, delta=JAR_DELTA):
    """Paper JAR: move +/-1 channel only if CQ improves by at least delta."""
    best = int(current_ch)
    best_cq = float(qv[best])
    for a in (best - 1, best + 1):
        if 0 <= a < len(qv):
            gain = float(qv[a]) - float(qv[current_ch])
            if gain >= delta and float(qv[a]) > best_cq:
                best, best_cq = a, float(qv[a])
    return best


def run_heuristic_episode(policy: str, n_nets: int, seed: int, number_of_channels=K):
    np.random.seed(int(seed))
    users, centers = set_random_location_of_networks(n_nets)
    env = python_env(
        number_of_nets=n_nets,
        number_of_users_in_each_net=users,
        net_center_location_and_std=centers,
        possible_channels=number_of_channels,
        add_noise=False,
        training=False,
    )
    obs, info = env.reset()
    game_history = []
    done = False
    while not done:
        agent_id = info["Master_Id"]
        time = info["Time"]
        current_ch = int(info["primaryChannel"])
        qv = np.asarray(obs, dtype=float)
        if policy == "ra":
            action = current_ch
        elif policy == "jar":
            action = jar_action(qv, current_ch)
        else:
            raise ValueError(policy)
        game_history.append([time, agent_id, action] + list(qv))
        obs, _, done, info = env.step(action, agent_id)
    return _metrics_from_history(game_history, n_nets, number_of_channels)


def run_drl_episode(weights, n_nets: int, seed: int, number_of_channels=K):
    np.random.seed(int(seed))
    users, centers = set_random_location_of_networks(n_nets)
    env = python_env(
        number_of_nets=n_nets,
        number_of_users_in_each_net=users,
        net_center_location_and_std=centers,
        possible_channels=number_of_channels,
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
    return _metrics_from_history(game_history, n_nets, number_of_channels)


def _metrics_from_history(game_history, n_nets, number_of_channels=K):
    gh_df = wrrape_game_history_do_df(game_history, number_of_channels)
    perf = get_game_performamce(
        game_history=gh_df,
        number_of_channels=number_of_channels,
        save_file=False,
    )
    (
        ancc,
        ct,
        cq_mean,
        cq_median,
        cq_max,
        cq_min,
        _above90,
        _below90,
        se,
        ancc_score,
        ct_score,
        ws,
        _n_used,
        _reuse,
    ) = perf
    return {
        "networks": int(n_nets),
        "cq_mean": float(cq_mean),
        "cq_min": float(cq_min),
        "cq_median": float(cq_median),
        "composite": float(0.5 * (cq_mean + cq_min)),  # paper Fig. 15 metric
        "ws": float(ws),
        "cts": float(ct_score),
        "anccs": float(ancc_score),
        "ses": float(se),
        "ancc": float(ancc),
    }


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

_WEIGHT_CACHE: dict[str, object] = {}


def _cached_weights(path: str):
    w = _WEIGHT_CACHE.get(path)
    if w is None:
        w = _load_weights(Path(path))
        _WEIGHT_CACHE[path] = w
    return w


def _job_heuristic(args):
    method, n_nets, game_idx, base_seed = args
    seed = int(base_seed) + 1000 * n_nets + game_idx
    return method, run_heuristic_episode(method, n_nets, seed)


def _job_drl(args):
    method, weights_path, n_nets, game_idx, base_seed = args
    weights = _cached_weights(weights_path)
    seed = int(base_seed) + 1000 * n_nets + game_idx
    return method, run_drl_episode(weights, n_nets, seed)


def build_jobs(weight_methods, games_per_n, base_seed, include_heuristics=True, n_range=None):
    n_range = list(n_range) if n_range is not None else list(N_RANGE)
    jobs = []
    for n in n_range:
        for g in range(games_per_n):
            if include_heuristics:
                jobs.append(("heuristic", ("ra", n, g, base_seed)))
                jobs.append(("heuristic", ("jar", n, g, base_seed)))
            for method, paths in weight_methods.items():
                # Round-robin across available trained seeds so Fig. 15
                # reflects multi-seed variability, not a single lucky seed.
                wp = paths[g % len(paths)]
                jobs.append(("drl", (method, str(wp), n, g, base_seed)))
    return jobs, n_range


def run_all(jobs, max_workers):
    """Run eval jobs sequentially (reliable under a concurrent training load).

    ``max_workers`` is accepted for API compatibility but ignored: each episode
    already saturates a core via NumPy/PyTorch, and process pools contend with
    the training campaign on Windows.
    """
    del max_workers  # unused on purpose
    rows = []
    ordered = [p for k, p in jobs if k == "heuristic"] + [p for k, p in jobs if k == "drl"]
    kinds = ["heuristic"] * sum(1 for k, _ in jobs if k == "heuristic") + [
        "drl"
    ] * sum(1 for k, _ in jobs if k == "drl")
    total = len(ordered)
    print(
        f"Running {total} eval games sequentially "
        f"({kinds.count('heuristic')} heuristic, {kinds.count('drl')} DRL)...",
        flush=True,
    )
    for i, (kind_payload, kind) in enumerate(zip(ordered, kinds), start=1):
        if kind == "heuristic":
            method, metrics = _job_heuristic(kind_payload)
        else:
            method, metrics = _job_drl(kind_payload)
        metrics["method"] = method
        rows.append(metrics)
        if i % 5 == 0 or i == total:
            print(
                f"  [{i}/{total}] {method} N={metrics['networks']} "
                f"composite={metrics['composite']:.3f}",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# Aggregation + plots (Fig. 15 + Fig. 13 style)
# ---------------------------------------------------------------------------

def aggregate(rows, n_range=None):
    n_range = list(n_range) if n_range is not None else list(N_RANGE)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    summary = {"methods": {}, "by_n": {}, "n_range": n_range}
    for method, rs in by_method.items():
        in_s = [r["composite"] for r in rs if r["networks"] <= IN_SAMPLE_MAX]
        out_s = [r["composite"] for r in rs if r["networks"] > IN_SAMPLE_MAX]
        all_s = [r["composite"] for r in rs]
        summary["methods"][method] = {
            "label": METHOD_LABELS.get(method, method),
            "n_games": len(rs),
            "composite_all": float(np.mean(all_s)),
            "composite_in": float(np.mean(in_s)) if in_s else float("nan"),
            "composite_out": float(np.mean(out_s)) if out_s else float("nan"),
            "ws_all": float(np.mean([r["ws"] for r in rs])),
            "cts_all": float(np.mean([r["cts"] for r in rs])),
        }
        per_n = {}
        for n in n_range:
            vals = [r["composite"] for r in rs if r["networks"] == n]
            if vals:
                per_n[str(n)] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        summary["by_n"][method] = per_n
    return summary


def plot_fig15_bars(summary, out_path: Path):
    """Grouped bars: in-sample vs out-of-sample, one group per method."""
    methods = [m for m in ("pytorch", "fedavg", "fedprox", "jar", "ra") if m in summary["methods"]]
    if not methods:
        return

    x = np.arange(2)
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for i, m in enumerate(methods):
        s = summary["methods"][m]
        vals = [s["composite_in"], s["composite_out"]]
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=METHOD_COLORS.get(m, "gray"),
            label=s["label"],
            edgecolor="white",
            linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r"In-sample" + "\n" + r"($\#$Networks $\leq 7$)",
         r"Out-of-sample" + "\n" + r"($\#$Networks $> 7$)"]
    )
    ax.set_ylabel(r"$E[(CQ + \min CQ)/2]$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.set_title("Algorithm comparison (paper Fig. 15 protocol)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def plot_fig13_curves(summary, out_path: Path):
    """E[(CQ+minCQ)/2] vs number of networks."""
    methods = [m for m in ("pytorch", "fedavg", "fedprox", "jar", "ra") if m in summary["by_n"]]
    if not methods:
        return
    n_range = summary.get("n_range", N_RANGE)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in methods:
        xs, ys, es = [], [], []
        for n in n_range:
            entry = summary["by_n"][m].get(str(n))
            if not entry:
                continue
            xs.append(n)
            ys.append(entry["mean"])
            es.append(entry["std"] / max(1, np.sqrt(entry["n"])))
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            ms=4,
            lw=1.6,
            capsize=2,
            color=METHOD_COLORS.get(m, "gray"),
            label=METHOD_LABELS.get(m, m),
        )
    ax.axvline(IN_SAMPLE_MAX + 0.5, color="0.5", ls="--", lw=0.9)
    ax.text(IN_SAMPLE_MAX - 0.3, 0.08, "in-sample", ha="right", fontsize=8, color="0.4")
    ax.text(IN_SAMPLE_MAX + 0.8, 0.08, "out-of-sample", ha="left", fontsize=8, color="0.4")
    ax.set_xlabel(r"$\#$ Networks")
    ax.set_ylabel(r"$E[(CQ + \min CQ)/2]$")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(n_range)
    ax.grid(alpha=0.3, lw=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Channel quality vs network count (paper Fig. 13 protocol)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--games-per-n", type=int, default=30, help="Paper uses 30")
    parser.add_argument("--max-seeds", type=int, default=None, help="Cap trained seeds used")
    parser.add_argument("--max-workers", type=int, default=1, help="Kept for compatibility; runs sequentially")
    parser.add_argument("--base-seed", type=int, default=424242)
    parser.add_argument("--skip-heuristics", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=["pytorch", "fedavg", "fedprox"], default=None)
    parser.add_argument("--n-list", nargs="+", type=int, default=None, help="Subset of N values (default 2..15)")
    parser.add_argument("--quick", action="store_true", help="Draft mode: N={2,4,6,8,10,12,15}, 3 games each")
    parser.add_argument("--out-subdir", default="paper_fig15", help="Output folder under the study dir")
    args = parser.parse_args(argv)

    study_dir = Path(args.study_dir).resolve()
    out_dir = study_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_range = [2, 4, 6, 8, 10, 12, 15]
        games_per_n = min(args.games_per_n, 3) if args.games_per_n != 30 else 3
    else:
        n_range = args.n_list if args.n_list else list(N_RANGE)
        games_per_n = args.games_per_n

    weight_methods = discover_weight_methods(study_dir, max_seeds=args.max_seeds)
    if args.methods:
        weight_methods = {k: v for k, v in weight_methods.items() if k in args.methods}
    print("Discovered weight files:")
    for m, paths in weight_methods.items():
        print(f"  {m}: {len(paths)} seeds (e.g. {paths[0].name})")

    jobs, n_range = build_jobs(
        weight_methods,
        games_per_n=games_per_n,
        base_seed=args.base_seed,
        include_heuristics=not args.skip_heuristics,
        n_range=n_range,
    )
    rows = run_all(jobs, max_workers=args.max_workers)

    raw_path = out_dir / "eval_rows.json"
    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", raw_path)

    summary = aggregate(rows, n_range=n_range)
    summary["meta"] = {
        "study_dir": str(study_dir),
        "games_per_n": games_per_n,
        "n_range": n_range,
        "total_games_per_method": games_per_n * len(n_range),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights_used": {m: len(p) for m, p in weight_methods.items()},
        "quick": bool(args.quick),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", summary_path)

    plot_fig15_bars(summary, out_dir / "fig15_composite_bars.png")
    plot_fig13_curves(summary, out_dir / "fig13_composite_vs_n.png")

    print("\n=== Fig. 15 summary: E[(CQ+minCQ)/2] ===")
    print(f"{'Method':20s} {'In-sample':>10s} {'Out-of-sample':>14s} {'All':>8s}")
    for m, s in summary["methods"].items():
        print(
            f"{s['label']:20s} {s['composite_in']:10.3f} {s['composite_out']:14.3f} {s['composite_all']:8.3f}"
        )


if __name__ == "__main__":
    main()
