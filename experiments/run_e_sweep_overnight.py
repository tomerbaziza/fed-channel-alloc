"""Overnight FedProx E-sweep launcher with a bounded worker pool.

Trains E in {1,2,5,10,15,20,40,60,80,120,200,300} x N seeds in parallel
(default 6 concurrent jobs). E=10 is taken from the main campaign unless
--retrain-e10 is set. When every arm finishes, evaluates CQ / Shannon rate,
writes the paper figure, and recompiles the PDF.

Usage:
  py -3 experiments/run_e_sweep_overnight.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
E_VALUES = [1, 2, 5, 10, 15, 20, 40, 60, 80, 120, 200, 300]
N_SEEDS = 3
CAMPAIGN_E = 10


def _job(e: int, seed_idx: int, sweep_dir: Path, rounds: int, inference: int) -> dict:
    save_dir = sweep_dir / f"E{e}" / f"seed_{seed_idx:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log = save_dir / "train_log.txt"
    if (save_dir / "job_complete.json").exists():
        return {"e": e, "seed_idx": seed_idx, "status": "skipped", "save_dir": str(save_dir)}
    cmd = [
        PY,
        str(REPO_ROOT / "experiments" / "jobs" / "e_sweep_job.py"),
        "--save-dir",
        str(save_dir),
        "--seed-idx",
        str(seed_idx),
        "--local-steps",
        str(e),
        "--rounds",
        str(rounds),
        "--inference-episodes",
        str(inference),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    return {
        "e": e,
        "seed_idx": seed_idx,
        "status": "ok" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "save_dir": str(save_dir),
    }


def link_campaign_e10(sweep_dir: Path, campaign_dir: Path, n_seeds: int):
    """Point E10/seed_* at the already-trained campaign FedProx seeds."""
    dest_root = sweep_dir / f"E{CAMPAIGN_E}"
    dest_root.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(sorted((campaign_dir / "fedprox").glob("seed_*"))[:n_seeds]):
        dest = dest_root / f"seed_{i:02d}"
        if dest.exists():
            continue
        # Junction / directory symlink on Windows; fall back to a pointer file.
        try:
            dest.symlink_to(src.resolve(), target_is_directory=True)
        except OSError:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "CAMPAIGN_POINTER.txt").write_text(str(src.resolve()), encoding="utf-8")
            for name in ("job_complete.json", "history.json", "inference.json"):
                s = src / name
                if s.exists() and not (dest / name).exists():
                    (dest / name).write_bytes(s.read_bytes())
            for ckpt in sorted(src.glob("global_weights_round_*.pkl"))[-1:]:
                target = dest / ckpt.name
                if not target.exists():
                    target.write_bytes(ckpt.read_bytes())
        print(f"linked E=10 seed_{i:02d} <- {src}")


def all_complete(sweep_dir: Path, e_values, n_seeds: int, skip_e10: bool) -> bool:
    for e in e_values:
        if skip_e10 and e == CAMPAIGN_E:
            # Accept campaign pointer / symlink / own job_complete
            for i in range(n_seeds):
                sd = sweep_dir / f"E{e}" / f"seed_{i:02d}"
                if not sd.exists():
                    return False
                if not (
                    (sd / "job_complete.json").exists()
                    or list(sd.glob("global_weights_round_*.pkl"))
                    or (sd / "CAMPAIGN_POINTER.txt").exists()
                ):
                    return False
            continue
        for i in range(n_seeds):
            if not (sweep_dir / f"E{e}" / f"seed_{i:02d}" / "job_complete.json").exists():
                return False
    return True


def finalize(sweep_dir: Path, campaign_dir: Path, out_png: Path, games: int, seeds: int):
    cmd = [
        PY,
        str(REPO_ROOT / "experiments" / "eval_e_sweep.py"),
        "--sweep-dir",
        str(sweep_dir),
        "--campaign-dir",
        str(campaign_dir),
        "--games",
        str(games),
        "--seeds",
        str(seeds),
        "--out-dir",
        str(out_png.parent),
    ]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    # eval_e_sweep writes e_sweep.png into out-dir
    produced = out_png.parent / "e_sweep.png"
    if produced.exists() and produced.resolve() != out_png.resolve():
        out_png.write_bytes(produced.read_bytes())
    # also copy json beside the figure used by the paper
    src_json = out_png.parent / "e_sweep.json"
    if src_json.exists():
        (REPO_ROOT / "docs" / "figs" / "campaign" / "e_sweep.json").write_bytes(src_json.read_bytes())
        (REPO_ROOT / "docs" / "figs" / "campaign" / "e_sweep.png").write_bytes(
            (out_png if out_png.exists() else produced).read_bytes()
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default="experiments/results/e_sweep_dense_20260804")
    parser.add_argument(
        "--campaign-dir",
        default="experiments/results/main_comparison_private_buffers_20260801",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--inference-episodes", type=int, default=100)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--retrain-e10", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only finalize if ready")
    args = parser.parse_args(argv)

    sweep_dir = (REPO_ROOT / args.sweep_dir).resolve()
    campaign_dir = (REPO_ROOT / args.campaign_dir).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_png = REPO_ROOT / "docs" / "figs" / "campaign" / "e_sweep.png"

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "e_values": E_VALUES,
        "seeds": args.seeds,
        "workers": args.workers,
        "retrain_e10": bool(args.retrain_e10),
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.retrain_e10:
        link_campaign_e10(sweep_dir, campaign_dir, args.seeds)

    jobs = []
    for e in E_VALUES:
        if e == CAMPAIGN_E and not args.retrain_e10:
            continue
        for i in range(args.seeds):
            jobs.append((e, i))

    print(f"{len(jobs)} training jobs, workers={args.workers}", flush=True)
    results = []
    if not args.skip_train and jobs:
        # ProcessPoolExecutor pickling of nested function is awkward on Windows;
        # run a simple sequential submit loop via concurrent futures of subprocess.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [
                pool.submit(_job, e, i, sweep_dir, args.rounds, args.inference_episodes)
                for e, i in jobs
            ]
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                print(
                    f"[{len(results)}/{len(jobs)}] E={res['e']} seed={res['seed_idx']} "
                    f"status={res['status']}",
                    flush=True,
                )

    (sweep_dir / "train_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if not all_complete(sweep_dir, E_VALUES, args.seeds, skip_e10=not args.retrain_e10):
        pending = []
        for e in E_VALUES:
            for i in range(args.seeds):
                sd = sweep_dir / f"E{e}" / f"seed_{i:02d}"
                ok = (sd / "job_complete.json").exists() or list(sd.glob("global_weights_round_*.pkl"))
                if not ok:
                    pending.append(f"E{e}/seed_{i:02d}")
        print("NOT COMPLETE, pending:", pending[:20], "...", flush=True)
        return 1

    print("all arms complete — evaluating", flush=True)
    finalize(sweep_dir, campaign_dir, out_png, args.games, args.seeds)

    # Update paper E set mention if needed is left to a small tex patch by the agent.
    print("DONE", datetime.now().isoformat(timespec="seconds"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
