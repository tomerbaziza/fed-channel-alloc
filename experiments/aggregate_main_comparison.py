"""Aggregate 30-seed main-comparison runs and append plots to theory doc."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_DOC = REPO_ROOT / "docs" / "federated_learning_theory.html"
CAMPAIGN_V2_START = "<!-- CAMPAIGN_RESULTS_V2_START -->"
CAMPAIGN_V2_END = "<!-- CAMPAIGN_RESULTS_V2_END -->"

METHOD_LABELS = {
    "pytorch": "PyTorch without FL (ρ=0.7)",
    "tf": "TF without FL (ρ=0.7)",
    "fedavg": "FedAvg best (E=10, ρ=1)",
    "fedprox": "FedProx best (E=10, μ=0.01, ρ=1)",
}

PYTORCH_MIN_TRAIN_REWARD = 70.0


def _smooth(values, window=20):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _x_for(values, window):
    if window <= 1 or len(values) < window:
        return np.arange(1, len(values) + 1)
    return np.arange(window, len(values) + 1)


def _select_seed_dirs(study_dir, method):
    study_dir = Path(study_dir)
    selected = []
    for seed_dir in sorted((study_dir / method).glob("seed_*")):
        if method != "pytorch":
            selected.append(seed_dir)
            continue
        jc_path = seed_dir / "job_complete.json"
        if not jc_path.is_file():
            continue
        try:
            jc = json.loads(jc_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if float(jc.get("last_20_train_reward", -1e9)) > PYTORCH_MIN_TRAIN_REWARD:
            selected.append(seed_dir)
    return selected


def _load_method_histories(study_dir, method, key="round_rewards", seed_dirs=None):
    study_dir = Path(study_dir)
    traces = []
    for seed_dir in sorted(seed_dirs if seed_dirs is not None else (study_dir / method).glob("seed_*")):
        hist_path = seed_dir / "history.json"
        if not hist_path.is_file():
            continue
        data = json.loads(hist_path.read_text(encoding="utf-8"))
        series = data.get(key) or []
        if len(series) >= 20:
            traces.append([float(v) for v in series])
    return traces


def _aggregate_traces(traces, smooth_window=20):
    if not traces:
        return None
    max_len = max(len(t) for t in traces)
    matrix = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        matrix[i, : len(t)] = t
    mean_raw = np.nanmean(matrix, axis=0)
    std_raw = np.nanstd(matrix, axis=0)
    mean_smooth = _smooth(mean_raw, window=smooth_window)
    std_smooth = _smooth(std_raw, window=smooth_window)
    x = _x_for(mean_raw, window=smooth_window)
    return {
        "x": x.tolist(),
        "mean": mean_smooth.tolist(),
        "std": std_smooth.tolist(),
        "n_seeds": len(traces),
        "last_20_mean": float(np.nanmean([np.mean(t[-20:]) for t in traces])),
        "last_20_std": float(np.nanstd([np.mean(t[-20:]) for t in traces])),
    }


def _aggregate_inference(study_dir, method, seed_dirs=None):
    vals = []
    for seed_dir in sorted(seed_dirs if seed_dirs is not None else (Path(study_dir) / method).glob("seed_*")):
        inf_path = seed_dir / "inference.json"
        if inf_path.is_file():
            inf = json.loads(inf_path.read_text(encoding="utf-8"))
        else:
            hist = seed_dir / "history.json"
            if not hist.is_file():
                continue
            inf = json.loads(hist.read_text(encoding="utf-8")).get("inference") or {}
        v = inf.get("inference_reward_last_20") or inf.get("inference_reward_mean")
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def aggregate_and_plot(study_dir, smooth_window=20):
    study_dir = Path(study_dir)
    plots_dir = study_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_agg = {}
    infer_agg = {}
    cc_agg = {}
    for method in METHOD_LABELS:
        seed_dirs = _select_seed_dirs(study_dir, method)
        train_agg[method] = _aggregate_traces(
            _load_method_histories(study_dir, method, "round_rewards", seed_dirs=seed_dirs),
            smooth_window=smooth_window,
        )
        infer_agg[method] = _aggregate_inference(study_dir, method, seed_dirs=seed_dirs)
        cc_agg[method] = _aggregate_traces(
            _load_method_histories(study_dir, method, "round_channel_changes", seed_dirs=seed_dirs),
            smooth_window=smooth_window,
        )

    fig, ax = plt.subplots(figsize=(11, 6))
    for method, label in METHOD_LABELS.items():
        agg = train_agg.get(method)
        if not agg:
            continue
        x = np.asarray(agg["x"])
        mean = np.asarray(agg["mean"])
        std = np.asarray(agg["std"])
        ax.plot(x, mean, label=f"{label} (n={agg['n_seeds']})", linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.axhline(88, color="gray", linestyle="--", alpha=0.4, label="Perfect game (~88)")
    ax.axhline(79, color="gray", linestyle=":", alpha=0.4, label="Paper train (~79)")
    ax.set_title("Main comparison — mean ± std over seeds (training reward, smoothed w=20)")
    ax.set_xlabel("Episode / Communication round")
    ax.set_ylabel("Average reward (smoothed)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    reward_plot = plots_dir / "main_four_methods_rho1_30avg.png"
    fig.savefig(reward_plot, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for method, label in METHOD_LABELS.items():
        agg = cc_agg.get(method)
        if not agg:
            continue
        ax.plot(np.asarray(agg["x"]), np.asarray(agg["mean"]), label=f"{label} (n={agg['n_seeds']})", linewidth=1.8)
    ax.set_title("Main comparison — channel changes (mean over seeds, smoothed w=20)")
    ax.set_xlabel("Episode / Communication round")
    ax.set_ylabel("Average channel changes (smoothed)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    cc_plot = plots_dir / "main_four_methods_cc_rho1_30avg.png"
    fig.savefig(cc_plot, dpi=160)
    plt.close(fig)

    # --- Inference bar chart ---
    inf_plot = plots_dir / "main_four_methods_inference_30avg.png"
    methods_with_inf = [m for m in METHOD_LABELS if infer_agg.get(m)]
    if methods_with_inf:
        labels = [METHOD_LABELS[m] for m in methods_with_inf]
        means = [infer_agg[m]["mean"] for m in methods_with_inf]
        stds = [infer_agg[m]["std"] for m in methods_with_inf]
        colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, means, yerr=stds, capsize=6,
                       color=colours[: len(methods_with_inf)], edgecolor="black", alpha=0.85)
        ax.axhline(88, color="gray", linestyle="--", alpha=0.4, label="Perfect game (~88)")
        ax.axhline(79, color="gray", linestyle=":", alpha=0.4, label="Paper train (~79)")
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                    f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Average inference reward")
        ax.set_title("Inference reward (100 episodes, frozen weights) — mean ± std over seeds")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(inf_plot, dpi=160)
        plt.close(fig)

    summary = {
        "study_dir": str(study_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "smooth_window": smooth_window,
        "filters": {"pytorch_min_train_reward": PYTORCH_MIN_TRAIN_REWARD},
        "train": {m: train_agg[m] for m in METHOD_LABELS if train_agg.get(m)},
        "inference": infer_agg,
        "plots": {"reward": str(reward_plot), "channel_changes": str(cc_plot), "inference": str(inf_plot)},
    }
    for method in list(summary["train"].keys()):
        entry = summary["train"][method]
        summary["train"][method] = {
            "n_seeds": entry["n_seeds"],
            "last_20_mean": entry["last_20_mean"],
            "last_20_std": entry["last_20_std"],
        }

    (study_dir / "aggregated_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _rel_from_docs(path):
    return Path(os.path.relpath(Path(path).resolve(), (REPO_ROOT / "docs").resolve())).as_posix()


def _html_table(summary):
    train_data = summary.get("train") or {}
    infer_data = summary.get("inference") or {}
    best_train = max(
        (train_data.get(m, {}).get("last_20_mean", -1) for m in METHOD_LABELS),
        default=-1,
    )
    best_infer = max(
        (infer_data.get(m, {}).get("mean", -1) for m in METHOD_LABELS),
        default=-1,
    )
    rows = []
    for method, label in METHOD_LABELS.items():
        train = train_data.get(method) or {}
        infer = infer_data.get(method) or {}
        if not train and not infer:
            continue
        train_val = train.get("last_20_mean")
        train_cell = "—"
        if train_val is not None:
            bold = " style='font-weight:bold;color:#2e7d32'" if train_val == best_train else ""
            train_cell = f"<span{bold}>{train_val:.2f}</span> ± {train['last_20_std']:.2f} (n={train['n_seeds']})"
        infer_val = infer.get("mean")
        infer_cell = "—"
        if infer_val is not None:
            bold = " style='font-weight:bold;color:#2e7d32'" if infer_val == best_infer else ""
            infer_cell = f"<span{bold}>{infer_val:.2f}</span> ± {infer['std']:.2f} (n={infer['n']})"
        rows.append(f"<tr><td>{label}</td><td>{train_cell}</td><td>{infer_cell}</td></tr>")
    if not rows:
        return "<p><em>No aggregated results yet.</em></p>"
    return (
        "<table><thead><tr><th>Method</th><th>Train reward (last-20 mean ± std)</th>"
        "<th>Inference reward (mean ± std)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def inject_into_theory(study_dir, summary):
    if not THEORY_DOC.is_file():
        return False
    text = THEORY_DOC.read_text(encoding="utf-8")
    if CAMPAIGN_V2_START not in text:
        marker = "  <!-- CAMPAIGN_RESULTS_END -->"
        block = (
            f"\n  {CAMPAIGN_V2_START}\n"
            "  <h3>15.8 Main comparison — ρ=1 FL + 30-seed average</h3>\n"
            "  <p class=\"campaign-meta\">Results pending…</p>\n"
            f"  {CAMPAIGN_V2_END}\n"
        )
        if marker not in text:
            return False
        text = text.replace(marker, block + "\n  " + marker)

    study_dir = Path(study_dir)
    reward_plot = study_dir / "plots" / "main_four_methods_rho1_30avg.png"
    cc_plot = study_dir / "plots" / "main_four_methods_cc_rho1_30avg.png"
    inf_plot = study_dir / "plots" / "main_four_methods_inference_30avg.png"
    n_seeds = next(iter(summary.get("train", {}).values()), {}).get("n_seeds", "?")

    reward_src = _rel_from_docs(reward_plot) if reward_plot.is_file() else ""
    cc_src = _rel_from_docs(cc_plot) if cc_plot.is_file() else ""
    inf_src = _rel_from_docs(inf_plot) if inf_plot.is_file() else ""

    section = f"""  <h3>15.8 Main comparison — ρ=1 FL + 30-seed average</h3>
  <p class="campaign-meta"><strong>Study:</strong> {study_dir.name} &nbsp;|&nbsp;
     <strong>Generated:</strong> {summary.get('generated_at', '—')} &nbsp;|&nbsp;
     <strong>Seeds per method:</strong> {n_seeds} &nbsp;|&nbsp;
     <strong>Train:</strong> 1000 rounds/ep &nbsp;|&nbsp; <strong>Inference:</strong> 100 episodes (no weight updates)</p>

  <div class="callout def">
    <strong>Settings</strong>
    FedAvg / FedProx use personal reward only (<code>ρ = 1</code>).
    PyTorch and TF centralized baselines keep paper <code>ρ = 0.7</code>.
    PyTorch display includes only seeds with train last-20 reward above <code>{summary.get('filters', {}).get('pytorch_min_train_reward', '70')}</code>.
    Each seed: full training then 100 evaluation episodes (ε=0, no backprop).
    Curves show mean ± std across seeds (smoothed window 20).
  </div>

  <h4>Aggregated metrics</h4>
  {_html_table(summary)}

  <h4>Training curves (mean ± std over seeds)</h4>
  <div class="diagram campaign-plot"><img src="{reward_src}" alt="Main comparison training reward"/></div>
  <div class="diagram campaign-plot"><img src="{cc_src}" alt="Main comparison channel changes"/></div>

  <h4>Inference reward (frozen weights, 100 episodes per seed)</h4>
  <div class="diagram campaign-plot"><img src="{inf_src}" alt="Inference reward comparison"/></div>

  <div class="callout good">
    <strong>Refresh</strong>
    <code>py -3 experiments/run_main_comparison_multiseed.py --study-dir experiments/results/{study_dir.name} --aggregate-only</code>
  </div>"""

    start = text.find(CAMPAIGN_V2_START)
    end = text.find(CAMPAIGN_V2_END)
    if start == -1 or end == -1:
        return False
    new_text = text[: start + len(CAMPAIGN_V2_START)] + "\n" + section + "\n  " + text[end:]
    THEORY_DOC.write_text(new_text, encoding="utf-8")
    return True
