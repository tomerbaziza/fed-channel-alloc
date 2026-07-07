"""Run parallel FedProx E/mu sweeps and final comparison runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fl_learning_check import assess_history, write_assessment
from main_v3 import run_federated_training

E_VALUES = [1, 10, 20, 40, 80, 160]
MU_VALUES = [0.0, 0.01, 0.1, 1.0]
SWEEP_MU = 0.01
SWEEP_E_FOR_MU = 1
COMMUNICATION_ROUNDS = 1000
BASE_SEED = 20260630


def _mu_tag(mu):
    return str(mu).replace(".", "p")


def _job(config):
    save_dir = Path(config["save_dir"])
    history_path = save_dir / "history.json"
    if history_path.is_file():
        return {"status": "skipped", "history_path": str(history_path), **config}

    history, _ = run_federated_training(
        communication_rounds=config["communication_rounds"],
        local_train_steps=config["local_train_steps"],
        fedprox_mu=config["fedprox_mu"],
        persistent_replay=True,
        replay_memory_size=10000,
        fixed_topology=config.get("fixed_topology", False),
        fixed_number_of_nets=config.get("fixed_number_of_nets", 6),
        scenario_map_seed=config.get("scenario_map_seed", 42),
        save_dir=str(save_dir),
        run_name=config["run_name"],
        seed=config["seed"],
        checkpoint_every=100,
    )
    write_assessment(history, save_dir / "assessment.json", min_rounds=20)
    return {"status": "completed", "history_path": str(history_path), **config}


def _score_history(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rewards = data.get("round_rewards", [])
    cc = data.get("round_channel_changes", [])
    if len(rewards) < 20:
        return float("-inf"), {}
    return float(np.mean(rewards[-20:])), {
        "reward_last_20": float(np.mean(rewards[-20:])),
        "cc_last_20": float(np.mean(cc[-20:])),
        "pct_of_88": 100.0 * float(np.mean(rewards[-20:])) / 88.0,
        "assessment": assess_history(data, min_rounds=20),
    }


def build_sweep_jobs(campaign_dir, communication_rounds):
    jobs = []
    for idx, e in enumerate(E_VALUES):
        jobs.append(
            {
                "phase": "e_sweep",
                "run_name": f"fedprox_E{e}_mu{SWEEP_MU}",
                "save_dir": str(campaign_dir / "sweeps" / f"fedprox_E{e}_mu{_mu_tag(SWEEP_MU)}"),
                "communication_rounds": communication_rounds,
                "local_train_steps": e,
                "fedprox_mu": SWEEP_MU,
                "seed": BASE_SEED + idx,
            }
        )
    for idx, mu in enumerate(MU_VALUES):
        jobs.append(
            {
                "phase": "mu_sweep",
                "run_name": f"fedprox_mu{mu}_E{SWEEP_E_FOR_MU}",
                "save_dir": str(campaign_dir / "sweeps" / f"fedprox_mu{_mu_tag(mu)}_E{SWEEP_E_FOR_MU}"),
                "communication_rounds": communication_rounds,
                "local_train_steps": SWEEP_E_FOR_MU,
                "fedprox_mu": float(mu),
                "seed": BASE_SEED + 100 + idx,
            }
        )
    return jobs


def run_parallel(jobs, max_workers):
    workers = max(1, min(int(max_workers), len(jobs)))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_job, job): job for job in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def pick_best_e(campaign_dir):
    scored = []
    for e in E_VALUES:
        path = campaign_dir / "sweeps" / f"fedprox_E{e}_mu{_mu_tag(SWEEP_MU)}" / "history.json"
        if not path.is_file():
            path = REPO_ROOT / "experiments/results/fedprox_E1_mu001_persistent/full/history.json" if e == 1 else path
        if path.is_file():
            score, metrics = _score_history(path)
            scored.append((score, e, str(path), metrics))
    if not scored:
        return 1, {}
    scored.sort(key=lambda x: x[0], reverse=True)
    best_e = scored[0][1]
    return best_e, {"best_e": best_e, "candidates": [{"E": e, "score": s, "metrics": m} for s, e, _, m in scored]}


def pick_best_mu(campaign_dir, best_e):
    scored = []
    for mu in MU_VALUES:
        if mu == 0.0:
            continue
        path = campaign_dir / "sweeps" / f"fedprox_mu{_mu_tag(mu)}_E{SWEEP_E_FOR_MU}" / "history.json"
        if path.is_file():
            score, metrics = _score_history(path)
            scored.append((score, mu, str(path), metrics))
    if not scored:
        return 0.01, {}
    scored.sort(key=lambda x: x[0], reverse=True)
    return float(scored[0][1]), {"best_mu": float(scored[0][1]), "candidates": [{"mu": mu, "score": s, "metrics": m} for s, mu, _, m in scored]}


def build_final_jobs(campaign_dir, best_e, best_mu, communication_rounds):
    return [
        {
            "phase": "final",
            "run_name": f"fedavg_E{best_e}_random",
            "save_dir": str(campaign_dir / "finals" / f"fedavg_E{best_e}_random"),
            "communication_rounds": communication_rounds,
            "local_train_steps": int(best_e),
            "fedprox_mu": 0.0,
            "fixed_topology": False,
            "seed": BASE_SEED + 500,
        },
        {
            "phase": "final",
            "run_name": f"fedavg_E{best_e}_fixed6",
            "save_dir": str(campaign_dir / "finals" / f"fedavg_E{best_e}_fixed6"),
            "communication_rounds": communication_rounds,
            "local_train_steps": int(best_e),
            "fedprox_mu": 0.0,
            "fixed_topology": True,
            "fixed_number_of_nets": 6,
            "scenario_map_seed": 42,
            "seed": BASE_SEED + 501,
        },
        {
            "phase": "final",
            "run_name": f"fedprox_E{best_e}_mu{best_mu}_random",
            "save_dir": str(campaign_dir / "finals" / f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_random"),
            "communication_rounds": communication_rounds,
            "local_train_steps": int(best_e),
            "fedprox_mu": float(best_mu),
            "fixed_topology": False,
            "seed": BASE_SEED + 502,
        },
        {
            "phase": "final",
            "run_name": f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_fixed6",
            "save_dir": str(campaign_dir / "finals" / f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_fixed6"),
            "communication_rounds": communication_rounds,
            "local_train_steps": int(best_e),
            "fedprox_mu": float(best_mu),
            "fixed_topology": True,
            "fixed_number_of_nets": 6,
            "scenario_map_seed": 42,
            "seed": BASE_SEED + 503,
        },
    ]


def import_existing_results(campaign_dir):
    imports = []
    existing_e1 = REPO_ROOT / "experiments/results/fedprox_E1_mu001_persistent/full/history.json"
    if existing_e1.is_file():
        dst = campaign_dir / "sweeps" / f"fedprox_E1_mu{_mu_tag(SWEEP_MU)}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing_e1, dst / "history.json")
        cfg_src = existing_e1.parent / "config.json"
        if cfg_src.is_file():
            shutil.copy2(cfg_src, dst / "config.json")
        imports.append(
            {
                "run_name": "fedprox_E1_mu0p01",
                "history_path": str(dst / "history.json"),
                "status": "imported",
                "phase": "e_sweep",
            }
        )
    return imports


def save_manifest(campaign_dir, payload):
    path = campaign_dir / "campaign_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir")
    parser.add_argument("--phase", choices=["sweeps", "finals", "all", "report"], default="all")
    parser.add_argument("--rounds", type=int, default=COMMUNICATION_ROUNDS)
    parser.add_argument("--max-workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 1)))
    args = parser.parse_args(argv)

    campaign_dir = Path(args.campaign_dir) if args.campaign_dir else (
        REPO_ROOT / "experiments" / "results" / f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "sweeps").mkdir(exist_ok=True)
    (campaign_dir / "finals").mkdir(exist_ok=True)

    manifest = {"campaign_dir": str(campaign_dir), "rounds": args.rounds, "imports": import_existing_results(campaign_dir)}

    if args.phase in ("sweeps", "all"):
        jobs = build_sweep_jobs(campaign_dir, args.rounds)
        manifest["sweep_results"] = run_parallel(jobs, max_workers=args.max_workers)
        best_e, best_e_info = pick_best_e(campaign_dir)
        manifest["best_e_selection"] = best_e_info
        manifest["best_e"] = best_e

    if args.phase in ("finals", "all"):
        if "best_e" not in manifest:
            manifest["best_e"], manifest["best_e_selection"] = pick_best_e(campaign_dir)
        best_e = manifest["best_e"]
        best_mu, best_mu_info = pick_best_mu(campaign_dir, best_e)
        manifest["best_mu_selection"] = best_mu_info
        manifest["best_mu"] = best_mu
        final_jobs = build_final_jobs(campaign_dir, best_e, best_mu, args.rounds)
        manifest["final_results"] = run_parallel(final_jobs, max_workers=min(4, args.max_workers))

    save_manifest(campaign_dir, manifest)

    if args.phase in ("report", "all"):
        from experiments.build_html_report import build_report

        report_path = build_report(campaign_dir)
        print(json.dumps({"campaign_dir": str(campaign_dir), "report": str(report_path)}, indent=2))
    else:
        print(json.dumps({"campaign_dir": str(campaign_dir), "manifest": str(campaign_dir / "campaign_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
