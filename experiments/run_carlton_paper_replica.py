"""Train a paper-faithful CARLTON CTDE replica (Table II) and compare to FL.

Locks the centralized DeepMellow loop to the published CARLTON settings:
  rho=0.7, Glorot-Uniform init, N in {2..7}, B=1000, N_E=40, bz=32,
  epsilon 0.5->0.01 over B/2, omega/lr schedule at B/2, gamma=0.9.

Runs a few seeds, then evaluates Fig.15-style composites and writes a
side-by-side comparison against the federated campaign summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_centralized import single_training_run

METHOD = "carlton_paper"
FL_SUMMARY_DEFAULT = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "main_comparison_private_buffers_20260801"
    / "paper_fig15"
    / "summary.json"
)


def _seed_dir(study_dir: Path, seed: int) -> Path:
    return study_dir / METHOD / f"seed_{int(seed):02d}"


def _is_complete(run_dir: Path) -> bool:
    return (run_dir / "job_complete.json").is_file() and (run_dir / "inference.json").is_file()


def _train_one(config: dict) -> dict:
    save_dir = Path(config["save_dir"]).resolve()
    if _is_complete(save_dir) and not config.get("force"):
        return {"status": "skipped", **config}
    save_dir.mkdir(parents=True, exist_ok=True)
    train_info, final, stopped, inference = single_training_run(
        n_episodes=config["episodes"],
        i_d_folder=f"paper_{config['seed']:02d}",
        fresh_start=True,
        early_check_at=0,
        log_every=50,
        checkpoint_every=0,
        rho=0.7,
        inference_episodes=config["inference_episodes"],
        seed=config["seed"],
        output_dir=str(save_dir),
    )
    rewards = list(train_info.average_accumulated_reward_vec)
    history = {
        "method": "CARLTON paper replica (PyTorch CTDE)",
        "config": {
            "episodes": config["episodes"],
            "inference_episodes": config["inference_episodes"],
            "seed": config["seed"],
            "rho": 0.7,
            "weight_init": "glorot",
            "n_train_range": [2, 7],
            "source": "arXiv:2402.17773 Table II",
        },
        "round_rewards": rewards,
        "round_channel_changes": list(train_info.average_changed_channels_vec),
        "inference": inference,
    }
    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if inference:
        (save_dir / "inference.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")
    payload = {
        "method": METHOD,
        "seed": config["seed"],
        "stopped_early": stopped,
        "final_status": final.get("status"),
        "last_20_train_reward": float(np.mean(rewards[-20:])) if rewards else None,
        "inference_last_20": (inference or {}).get("inference_reward_last_20"),
        "inference_mean": (inference or {}).get("inference_reward_mean"),
    }
    (save_dir / "job_complete.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"status": "completed", **payload}


def _aggregate_rewards(study_dir: Path, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        p = _seed_dir(study_dir, seed) / "job_complete.json"
        if not p.is_file():
            continue
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        return {}
    train = [r["last_20_train_reward"] for r in rows if r.get("last_20_train_reward") is not None]
    inf = [r["inference_last_20"] for r in rows if r.get("inference_last_20") is not None]
    return {
        "n_seeds": len(rows),
        "train_last20_mean": float(np.mean(train)) if train else None,
        "train_last20_std": float(np.std(train)) if train else None,
        "inference_last20_mean": float(np.mean(inf)) if inf else None,
        "inference_last20_std": float(np.std(inf)) if inf else None,
        "per_seed": rows,
    }


def _run_fig15(study_dir: Path, games_per_n: int, max_seeds: int | None) -> Path:
    from experiments.eval_paper_fig15 import (
        aggregate,
        build_jobs,
        plot_fig13_curves,
        plot_fig15_bars,
        run_all,
        _latest_pytorch_weights,
    )

    roots = sorted((study_dir / METHOD).glob("seed_*"))
    if max_seeds is not None:
        roots = roots[:max_seeds]
    paths = []
    for sd in roots:
        p = _latest_pytorch_weights(sd)
        if p is not None:
            paths.append(p)
    if not paths:
        raise RuntimeError(f"No trained weights found under {study_dir / METHOD}")

    jobs, n_range = build_jobs(
        {METHOD: paths},
        games_per_n=games_per_n,
        base_seed=12345,
        include_heuristics=True,
    )
    rows = run_all(jobs, max_workers=1)
    summary = aggregate(rows, n_range=n_range)
    summary["meta"] = {
        "study_dir": str(study_dir),
        "method": METHOD,
        "games_per_n": games_per_n,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = study_dir / "paper_fig15"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_fig15_bars(summary, out_dir / "fig15_composite_bars.png")
    plot_fig13_curves(summary, out_dir / "fig13_composite_vs_n.png")
    return summary_path


def _compare(fl_summary_path: Path, replica_summary_path: Path, reward_agg: dict) -> dict:
    fl = json.loads(fl_summary_path.read_text(encoding="utf-8")) if fl_summary_path.is_file() else {}
    rep = json.loads(replica_summary_path.read_text(encoding="utf-8"))
    fl_methods = fl.get("methods", {})
    rep_m = rep["methods"].get(METHOD, {})

    def row(name, m):
        return {
            "composite_all": m.get("composite_all"),
            "composite_in": m.get("composite_in"),
            "composite_out": m.get("composite_out"),
            "ws_all": m.get("ws_all"),
        }

    comparison = {
        "carlton_paper_replica": {
            "rewards": reward_agg,
            "fig15": row(METHOD, rep_m),
        },
        "federated_campaign_fig15": {
            k: row(k, fl_methods[k])
            for k in ("pytorch", "fedavg", "fedprox", "jar", "ra")
            if k in fl_methods
        },
        "verdict_note": (
            "Compare Fig.15 composites primarily. Reward scales differ when "
            "rho differs (paper replica uses rho=0.7; FL campaign used rho=1)."
        ),
    }
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--inference-episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--games-per-n", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-fig15", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--study-dir",
        default="",
        help="Output study directory (default: experiments/results/carlton_paper_replica_<date>)",
    )
    parser.add_argument("--fl-summary", default=str(FL_SUMMARY_DEFAULT))
    args = parser.parse_args()

    study_dir = Path(args.study_dir) if args.study_dir else (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"carlton_paper_replica_{datetime.now().strftime('%Y%m%d')}"
    )
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "manifest.json").write_text(
        json.dumps(
            {
                "method": METHOD,
                "seeds": args.seeds,
                "episodes": args.episodes,
                "rho": 0.7,
                "weight_init": "glorot",
                "n_train": "2..7 inclusive",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    jobs = [
        {
            "seed": int(s),
            "episodes": int(args.episodes),
            "inference_episodes": int(args.inference_episodes),
            "save_dir": str(_seed_dir(study_dir, s)),
            "force": bool(args.force),
        }
        for s in args.seeds
    ]

    if not args.skip_train:
        print(
            f"Training CARLTON paper replica: {len(jobs)} seeds x {args.episodes} episodes "
            f"(workers={args.workers}) -> {study_dir}",
            flush=True,
        )
        if args.workers <= 1:
            for job in jobs:
                print(_train_one(job), flush=True)
        else:
            # Windows-safe: each worker is a fresh process.
            with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                futs = {ex.submit(_train_one, job): job["seed"] for job in jobs}
                for fut in as_completed(futs):
                    seed = futs[fut]
                    try:
                        res = fut.result()
                        print(f"seed {seed}: {res.get('status')} "
                              f"train20={res.get('last_20_train_reward')} "
                              f"inf20={res.get('inference_last_20')}", flush=True)
                    except Exception as exc:
                        print(f"seed {seed} FAILED: {exc}", flush=True)
                        raise

    reward_agg = _aggregate_rewards(study_dir, list(args.seeds))
    (study_dir / "reward_summary.json").write_text(
        json.dumps(reward_agg, indent=2), encoding="utf-8"
    )
    print("Reward summary:", json.dumps(reward_agg, indent=2), flush=True)

    replica_summary = study_dir / "paper_fig15" / "summary.json"
    if not args.skip_fig15:
        print("Running Fig.15-style evaluation...", flush=True)
        replica_summary = _run_fig15(study_dir, args.games_per_n, max_seeds=len(args.seeds))
        print(f"Wrote {replica_summary}", flush=True)

    if replica_summary.is_file():
        comparison = _compare(Path(args.fl_summary), replica_summary, reward_agg)
        out = study_dir / "comparison_vs_fl.json"
        out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print("\n=== Comparison vs FL campaign (Fig.15 composite_all) ===", flush=True)
        print(
            f"  CARLTON paper replica: {comparison['carlton_paper_replica']['fig15'].get('composite_all')}",
            flush=True,
        )
        for k, v in comparison["federated_campaign_fig15"].items():
            print(f"  {k:12s}: {v.get('composite_all')}", flush=True)
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    # Required for Windows ProcessPoolExecutor spawn.
    main()
