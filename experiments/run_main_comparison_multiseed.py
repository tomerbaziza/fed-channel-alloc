"""30-seed main comparison: TF / PyTorch / FedAvg / FedProx + 100 inference episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_centralized import single_training_run
from main_v3 import run_federated_training

N_SEEDS = 30
TRAIN_ROUNDS = 1000
INFERENCE_EPISODES = 100
BEST_E = 10
BEST_MU = 0.01
BASE_SEED = 20260701


METHODS = (
    "pytorch",
    "tf",
    "fedavg",
    "fedprox",
)


def _seed_dir(study_dir, method, seed_idx):
    return Path(study_dir).resolve() / method / f"seed_{seed_idx:02d}"


def _is_complete(run_dir):
    run_dir = Path(run_dir)
    return (run_dir / "job_complete.json").is_file() or (
        (run_dir / "history.json").is_file() and (run_dir / "inference.json").is_file()
    )


def _write_job_complete(run_dir, payload):
    path = Path(run_dir) / "job_complete.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _job_pytorch(config):
    save_dir = Path(config["save_dir"]).resolve()
    if _is_complete(save_dir):
        return {"status": "skipped", **config}
    save_dir.mkdir(parents=True, exist_ok=True)
    train_info, final, stopped, inference = single_training_run(
        n_episodes=config["episodes"],
        i_d_folder=f"mc_{config['seed']:02d}",
        fresh_start=True,
        early_check_at=0,
        log_every=100,
        checkpoint_every=0,
        rho=0.7,
        inference_episodes=config["inference_episodes"],
        seed=config["seed"],
        output_dir=str(save_dir),
    )
    history = {
        "method": "PyTorch centralized (no FL)",
        "config": {
            "episodes": config["episodes"],
            "inference_episodes": config["inference_episodes"],
            "seed": config["seed"],
            "rho": 0.7,
        },
        "round_rewards": list(train_info.average_accumulated_reward_vec),
        "round_channel_changes": list(train_info.average_changed_channels_vec),
        "inference": inference,
    }
    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if inference:
        (save_dir / "inference.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")
    _write_job_complete(
        save_dir,
        {
            "method": "pytorch",
            "seed": config["seed"],
            "stopped_early": stopped,
            "final_status": final.get("status"),
            "last_20_train_reward": float(np.mean(history["round_rewards"][-20:])),
            "inference_last_20": (inference or {}).get("inference_reward_last_20"),
        },
    )
    return {"status": "completed", **config}


def _tf_executable():
    import shutil

    if shutil.which("py"):
        return ["py", "-3.11"]
    return [sys.executable]


def _job_tf(config):
    save_dir = Path(config["save_dir"]).resolve()
    if _is_complete(save_dir):
        return {"status": "skipped", **config}
    cmd = _tf_executable() + [
        str(REPO_ROOT / "experiments" / "jobs" / "tf_centralized_job.py"),
        "--save-dir",
        str(save_dir),
        "--seed",
        str(config["seed"]),
        "--episodes",
        str(config["episodes"]),
        "--inference-episodes",
        str(config["inference_episodes"]),
    ]
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"TF job failed seed={config['seed']}:\n{proc.stderr}\n{proc.stdout}")
    return {"status": "completed", **config}


def _job_federated(config):
    save_dir = Path(config["save_dir"]).resolve()
    if _is_complete(save_dir):
        return {"status": "skipped", **config}
    save_dir.mkdir(parents=True, exist_ok=True)
    history, _weights = run_federated_training(
        communication_rounds=config["episodes"],
        local_train_steps=config["local_train_steps"],
        fedprox_mu=config["fedprox_mu"],
        persistent_replay=True,
        replay_memory_size=10000,
        fixed_topology=False,
        save_dir=str(save_dir),
        run_name=config["run_name"],
        seed=config["seed"],
        checkpoint_every=100,
        rho=1.0,
        inference_episodes=config["inference_episodes"],
    )
    _write_job_complete(
        save_dir,
        {
            "method": config["method"],
            "seed": config["seed"],
            "last_20_train_reward": float(np.mean(history["round_rewards"][-20:])),
            "inference_last_20": (history.get("inference") or {}).get("inference_reward_last_20"),
        },
    )
    return {"status": "completed", **config}


def build_jobs(study_dir, n_seeds=N_SEEDS):
    jobs = []
    for idx in range(int(n_seeds)):
        seed = BASE_SEED + idx
        jobs.append(
            {
                "method": "pytorch",
                "save_dir": str(_seed_dir(study_dir, "pytorch", idx)),
                "seed": seed,
                "episodes": TRAIN_ROUNDS,
                "inference_episodes": INFERENCE_EPISODES,
            }
        )
        jobs.append(
            {
                "method": "tf",
                "save_dir": str(_seed_dir(study_dir, "tf", idx)),
                "seed": seed,
                "episodes": TRAIN_ROUNDS,
                "inference_episodes": INFERENCE_EPISODES,
            }
        )
        jobs.append(
            {
                "method": "fedavg",
                "save_dir": str(_seed_dir(study_dir, "fedavg", idx)),
                "seed": seed + 200,
                "episodes": TRAIN_ROUNDS,
                "inference_episodes": INFERENCE_EPISODES,
                "local_train_steps": BEST_E,
                "fedprox_mu": 0.0,
                "run_name": f"fedavg_E{BEST_E}_seed{idx:02d}",
            }
        )
        jobs.append(
            {
                "method": "fedprox",
                "save_dir": str(_seed_dir(study_dir, "fedprox", idx)),
                "seed": seed + 400,
                "episodes": TRAIN_ROUNDS,
                "inference_episodes": INFERENCE_EPISODES,
                "local_train_steps": BEST_E,
                "fedprox_mu": BEST_MU,
                "run_name": f"fedprox_E{BEST_E}_mu{BEST_MU}_seed{idx:02d}",
            }
        )
    return jobs


def _dispatch(job):
    method = job["method"]
    if method == "pytorch":
        return _job_pytorch(job)
    if method == "tf":
        return _job_tf(job)
    if method in ("fedavg", "fedprox"):
        return _job_federated(job)
    raise ValueError(method)


def run_jobs(jobs, max_workers):
    workers = max(1, min(int(max_workers), len(jobs)))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_dispatch, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"status": "error", "error": str(exc), **job})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--max-workers", type=int, default=max(1, min(2, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)

    study_dir = Path(args.study_dir).resolve() if args.study_dir else (
        REPO_ROOT / "experiments" / "results" / f"main_comparison_rho1_multiseed_{datetime.now().strftime('%Y%m%d')}"
    ).resolve()
    study_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        all_jobs = build_jobs(study_dir, n_seeds=args.n_seeds)
        jobs = [j for j in all_jobs if j["method"] in args.methods]
        results = run_jobs(jobs, max_workers=args.max_workers)
        manifest = {
            "study_dir": str(study_dir),
            "n_seeds": args.n_seeds,
            "train_rounds": TRAIN_ROUNDS,
            "inference_episodes": INFERENCE_EPISODES,
            "fedavg_fedprox_rho": 1.0,
            "centralized_rho": 0.7,
            "best_e": BEST_E,
            "best_mu": BEST_MU,
            "results": results,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    from experiments.aggregate_main_comparison import aggregate_and_plot, inject_into_theory

    summary = aggregate_and_plot(study_dir)
    inject_into_theory(study_dir, summary)
    print(json.dumps({"study_dir": str(study_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
