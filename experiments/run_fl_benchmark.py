"""Orchestrate FedAvg/FedProx CARLTON benchmark runs."""

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

from experiments.fl_learning_check import assess_history, write_assessment
from experiments.parse_training_logs import run_parser_preflight
from experiments.plot_fl_benchmark import main as plot_main
from main_v3 import run_federated_training


E_VALUES = [1, 2, 5, 10, 20, 40, 80]
MU_VALUES = [0.0, 0.01, 0.1, 1.0]
MU_LOW = 0.01
MU_HIGH = 1.0
COMMUNICATION_ROUNDS = 1000
SMOKE_ROUNDS = 30
SMOKE_CHECK_AT = 20
BASE_SEED = 20260630


def _status_path(run_dir, phase):
    return Path(run_dir) / f"{phase}_status.json"


def write_phase_status(run_dir, phase, status, **extra):
    payload = {
        "phase": phase,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    path = _status_path(run_dir, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def phase_completed(run_dir, phase):
    path = _status_path(run_dir, phase)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def _run_command(command):
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(command)
            + "\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )
    return completed.stdout.strip()


def run_preflight():
    commands = [
        [sys.executable, "test_federated_run.py"],
        [sys.executable, "test_fedprox_run.py"],
        [sys.executable, "test_fedprox_mu_zero.py"],
    ]
    outputs = {str(command): _run_command(command) for command in commands}
    parser_report = run_parser_preflight()
    plot_output = _run_command(
        [sys.executable, "experiments/plot_fl_benchmark.py", "--dry-run", "--validate"]
    )
    return {
        "unit_tests": outputs,
        "parser": parser_report,
        "plot_dry_run": json.loads(plot_output),
    }


def _mu_tag(mu):
    return str(mu).replace(".", "p")


def _training_job(config):
    save_dir = Path(config["save_dir"])
    complete_path = save_dir / "history.json"
    if complete_path.is_file():
        return {"status": "skipped", "history_path": str(complete_path), **config}

    history, _ = run_federated_training(
        communication_rounds=config["communication_rounds"],
        local_train_steps=config["local_train_steps"],
        fedprox_mu=config["fedprox_mu"],
        save_dir=str(save_dir),
        run_name=config["run_name"],
        seed=config["seed"],
        checkpoint_every=config.get("checkpoint_every", 100),
    )
    write_assessment(history, save_dir / "assessment.json", min_rounds=min(20, len(history["round_rewards"])))
    return {"status": "completed", "history_path": str(complete_path), **config}


def run_smoke(run_dir):
    save_dir = Path(run_dir) / "smoke_fedavg_E5"
    history, _ = run_federated_training(
        communication_rounds=SMOKE_ROUNDS,
        local_train_steps=5,
        fedprox_mu=0.0,
        save_dir=str(save_dir),
        run_name="smoke_fedavg_E5",
        seed=BASE_SEED,
        checkpoint_every=10,
        early_check_at=SMOKE_CHECK_AT,
        early_check_callback=lambda history: assess_history(history, min_rounds=SMOKE_CHECK_AT),
    )
    assessment = write_assessment(history, save_dir / "assessment.json", min_rounds=SMOKE_CHECK_AT)
    if assessment["status"] == "fail":
        write_phase_status(run_dir, "smoke", "failed", assessment=assessment)
        raise RuntimeError(f"Smoke failed: {assessment}")
    write_phase_status(run_dir, "smoke", "completed", assessment=assessment, history_path=str(save_dir / "history.json"))
    return assessment


def _parallel_jobs(configs, max_workers):
    workers = max(1, min(int(max_workers), len(configs)))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_config = {executor.submit(_training_job, config): config for config in configs}
        for future in as_completed(future_to_config):
            results.append(future.result())
    return results


def run_fedavg_sweep(run_dir, max_workers):
    configs = []
    for idx, e_value in enumerate(E_VALUES):
        configs.append(
            {
                "run_name": f"fedavg_E{e_value}",
                "save_dir": str(Path(run_dir) / f"fedavg_E{e_value}"),
                "communication_rounds": COMMUNICATION_ROUNDS,
                "local_train_steps": e_value,
                "fedprox_mu": 0.0,
                "seed": BASE_SEED + idx,
                "checkpoint_every": 100,
            }
        )
    results = _parallel_jobs(configs, max_workers=max_workers)

    scored = []
    for result in results:
        history = json.loads(Path(result["history_path"]).read_text(encoding="utf-8"))
        score = float(np.mean(history["round_rewards"][-20:]))
        scored.append((score, int(history["config"]["local_train_steps"]), result["history_path"]))

    best_score, best_e, history_path = max(scored, key=lambda item: item[0])
    best = {"best_e": best_e, "score_last_20_reward": best_score, "history_path": history_path}
    (Path(run_dir) / "best_e.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_phase_status(run_dir, "fedavg_sweep", "completed", best=best, runs=results)
    plot_main(["--results-dir", str(run_dir), "--plots-dir", str(Path(run_dir) / "plots"), "--phase", "fedavg", "--validate"])
    return best


def run_fedprox_sweep(run_dir, best_e, max_workers):
    configs = []
    for idx, mu in enumerate(MU_VALUES):
        configs.append(
            {
                "run_name": f"fedprox_mu{mu}",
                "save_dir": str(Path(run_dir) / f"fedprox_mu{_mu_tag(mu)}"),
                "communication_rounds": COMMUNICATION_ROUNDS,
                "local_train_steps": int(best_e),
                "fedprox_mu": float(mu),
                "seed": BASE_SEED + 100 + idx,
                "checkpoint_every": 100,
            }
        )
    results = _parallel_jobs(configs, max_workers=max_workers)
    write_phase_status(run_dir, "fedprox_sweep", "completed", best_e=best_e, runs=results)
    plot_main(["--results-dir", str(run_dir), "--plots-dir", str(Path(run_dir) / "plots"), "--phase", "fedprox", "--validate"])
    return results


def run_main_plot(run_dir):
    plot_main(["--results-dir", str(run_dir), "--plots-dir", str(Path(run_dir) / "plots"), "--phase", "main", "--validate"])
    write_phase_status(run_dir, "main_plot", "completed", plots_dir=str(Path(run_dir) / "plots"))


def create_run_dir(results_root):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_all(run_dir, max_workers):
    if not phase_completed(run_dir, "preflight"):
        preflight = run_preflight()
        write_phase_status(run_dir, "preflight", "completed", report=preflight)
    if not phase_completed(run_dir, "smoke"):
        run_smoke(run_dir)

    best_path = Path(run_dir) / "best_e.json"
    if phase_completed(run_dir, "fedavg_sweep") and best_path.is_file():
        best = json.loads(best_path.read_text(encoding="utf-8"))
    else:
        best = run_fedavg_sweep(run_dir, max_workers=max_workers)

    if not phase_completed(run_dir, "fedprox_sweep"):
        run_fedprox_sweep(run_dir, best["best_e"], max_workers=max_workers)
    if not phase_completed(run_dir, "main_plot"):
        run_main_plot(run_dir)
    write_phase_status(run_dir, "all", "completed")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--results-root", default=str(REPO_ROOT / "experiments" / "results"))
    parser.add_argument("--max-workers", type=int, default=max(1, min(7, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args(argv)

    run_dir = Path(args.resume) if args.resume else create_run_dir(args.results_root)
    if args.preflight:
        report = run_preflight()
        write_phase_status(run_dir, "preflight", "completed", report=report)
        print(json.dumps({"run_dir": str(run_dir), "preflight": report}, indent=2))
        return
    if args.smoke:
        assessment = run_smoke(run_dir)
        print(json.dumps({"run_dir": str(run_dir), "smoke": assessment}, indent=2))
        return
    if args.all:
        completed = run_all(run_dir, max_workers=args.max_workers)
        print(json.dumps({"run_dir": str(completed), "status": "completed"}, indent=2))
        return
    parser.error("Choose one of --preflight, --smoke, or --all.")


if __name__ == "__main__":
    main()
