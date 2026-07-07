"""Plot FedAvg/FedProx sweeps and centralized-vs-FL comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from parse_training_logs import load_centralized_traces, load_federated_history
except ImportError:  # pragma: no cover - supports package-style imports.
    from experiments.parse_training_logs import load_centralized_traces, load_federated_history


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_RESULTS_DIR = EXPERIMENTS_DIR / "results"
DEFAULT_PLOTS_DIR = EXPERIMENTS_DIR / "plots"


def smooth(values, window=20):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _x_for(values, window):
    if window <= 1 or len(values) < window:
        return np.arange(1, len(values) + 1)
    return np.arange(window, len(values) + 1)


def _plot_traces(traces, title, output_path, expected_lines=None, smooth_window=20):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, trace in traces.items():
        rewards = trace["rewards"]
        y = smooth(rewards, window=smooth_window)
        x = _x_for(rewards, window=smooth_window)
        ax.plot(x, y, label=label, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Episode / Communication round")
    ax.set_ylabel("Average reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    line_count = len(ax.lines)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    if expected_lines is not None and line_count != expected_lines:
        raise ValueError(f"{output_path.name}: expected {expected_lines} lines, got {line_count}")
    return {"path": str(output_path), "line_count": line_count}


def _history_files(results_dir, prefix):
    return sorted(Path(results_dir).glob(f"{prefix}*/history.json"))


def _label_from_history(path, kind):
    trace = load_federated_history(path)
    config = trace.get("config", {})
    if kind == "fedavg":
        return f"E={config.get('local_train_steps', path.parent.name)}", trace
    if kind == "fedprox":
        return f"mu={config.get('fedprox_mu', path.parent.name)}", trace
    return trace["label"], trace


def plot_fedavg_sweep(results_dir, plots_dir, smooth_window=20):
    traces = dict(_label_from_history(path, "fedavg") for path in _history_files(results_dir, "fedavg_E"))
    if not traces:
        raise FileNotFoundError(f"No fedavg_E*/history.json files found under {results_dir}")
    return _plot_traces(
        traces,
        "FedAvg E sweep",
        Path(plots_dir) / "fedavg_e_sweep.png",
        expected_lines=len(traces),
        smooth_window=smooth_window,
    )


def plot_fedprox_sweep(results_dir, plots_dir, smooth_window=20):
    traces = dict(_label_from_history(path, "fedprox") for path in _history_files(results_dir, "fedprox_mu"))
    if not traces:
        raise FileNotFoundError(f"No fedprox_mu*/history.json files found under {results_dir}")
    return _plot_traces(
        traces,
        "FedProx mu sweep (mu=0 is FedAvg)",
        Path(plots_dir) / "fedprox_mu_sweep.png",
        expected_lines=len(traces),
        smooth_window=smooth_window,
    )


def _best_fedavg_history(results_dir):
    best_path = Path(results_dir) / "best_e.json"
    if best_path.is_file():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        return Path(best["history_path"])

    candidates = []
    for path in _history_files(results_dir, "fedavg_E"):
        trace = load_federated_history(path)
        score = float(np.mean(trace["rewards"][-20:]))
        candidates.append((score, path))
    if not candidates:
        raise FileNotFoundError(f"No FedAvg histories found under {results_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _fedprox_history_for_mu(results_dir, mu):
    wanted = float(mu)
    for path in _history_files(results_dir, "fedprox_mu"):
        trace = load_federated_history(path)
        if abs(float(trace.get("config", {}).get("fedprox_mu", -1.0)) - wanted) < 1e-12:
            return path
    raise FileNotFoundError(f"No FedProx history found for mu={mu} under {results_dir}")


def plot_main_comparison(results_dir, plots_dir, smooth_window=20, mu_low=0.01, mu_high=1.0):
    traces = load_centralized_traces(min_points=10)
    fedavg_path = _best_fedavg_history(results_dir)
    fedavg_trace = load_federated_history(fedavg_path)
    best_e = fedavg_trace.get("config", {}).get("local_train_steps")
    traces[f"FedAvg best E={best_e}"] = fedavg_trace

    for label, mu in ((f"FedProx mu low={mu_low}", mu_low), (f"FedProx mu high={mu_high}", mu_high)):
        traces[label] = load_federated_history(_fedprox_history_for_mu(results_dir, mu))

    return _plot_traces(
        traces,
        "Reward comparison: centralized baselines vs FL",
        Path(plots_dir) / "main_comparison.png",
        expected_lines=5,
        smooth_window=smooth_window,
    )


def create_dry_run_results(results_dir):
    results_dir = Path(results_dir)
    rng = np.random.default_rng(7)
    rounds = 50
    base = np.linspace(45.0, 70.0, rounds)

    def write_history(folder, run_name, local_train_steps, fedprox_mu, offset):
        folder.mkdir(parents=True, exist_ok=True)
        rewards = base + offset + rng.normal(0.0, 1.5, rounds)
        history = {
            "run_name": run_name,
            "config": {
                "local_train_steps": local_train_steps,
                "fedprox_mu": fedprox_mu,
                "communication_rounds": rounds,
            },
            "round_rewards": rewards.tolist(),
            "round_channel_changes": np.linspace(12.0, 4.0, rounds).tolist(),
        }
        (folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return folder / "history.json"

    for idx, e_value in enumerate([1, 2, 5, 10, 20, 40, 80]):
        path = write_history(results_dir / f"fedavg_E{e_value}", f"fedavg_E{e_value}", e_value, 0.0, idx)
        if e_value == 20:
            (results_dir / "best_e.json").write_text(
                json.dumps({"best_e": 20, "history_path": str(path)}, indent=2),
                encoding="utf-8",
            )

    for idx, mu in enumerate([0.0, 0.01, 0.1, 1.0]):
        mu_tag = str(mu).replace(".", "p")
        write_history(results_dir / f"fedprox_mu{mu_tag}", f"fedprox_mu{mu}", 20, mu, idx * 1.5)


def validate_plot_outputs(plots_dir, expected=None):
    expected = expected or {
        "fedavg_e_sweep.png": 7,
        "fedprox_mu_sweep.png": 4,
        "main_comparison.png": 5,
    }
    plots_dir = Path(plots_dir)
    manifest_path = plots_dir / "plot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for filename, expected_lines in expected.items():
        path = plots_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size <= 5_000:
            raise ValueError(f"{path} is too small to be a valid plot.")
        recorded = manifest.get(filename, {}).get("line_count")
        if recorded is not None and int(recorded) != int(expected_lines):
            raise ValueError(f"{filename}: expected {expected_lines} lines, got {recorded}")
    return True


def write_manifest(plots_dir, entries):
    manifest = {Path(entry["path"]).name: entry for entry in entries}
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    (Path(plots_dir) / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR))
    parser.add_argument("--phase", choices=["all", "fedavg", "fedprox", "main"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=20)
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    if args.dry_run:
        results_dir = DEFAULT_RESULTS_DIR / "_dry_run"
        plots_dir = DEFAULT_PLOTS_DIR / "_dry_run"
        create_dry_run_results(results_dir)

    entries = []
    if args.phase in ("all", "fedavg"):
        entries.append(plot_fedavg_sweep(results_dir, plots_dir, smooth_window=args.smooth_window))
    if args.phase in ("all", "fedprox"):
        entries.append(plot_fedprox_sweep(results_dir, plots_dir, smooth_window=args.smooth_window))
    if args.phase in ("all", "main"):
        entries.append(plot_main_comparison(results_dir, plots_dir, smooth_window=args.smooth_window))
    write_manifest(plots_dir, entries)

    if args.validate:
        expected = {
            "fedavg_e_sweep.png": 7,
            "fedprox_mu_sweep.png": 4,
            "main_comparison.png": 5,
        }
        if args.phase == "fedavg":
            expected = {"fedavg_e_sweep.png": entries[0]["line_count"]}
        elif args.phase == "fedprox":
            expected = {"fedprox_mu_sweep.png": entries[0]["line_count"]}
        elif args.phase == "main":
            expected = {"main_comparison.png": 5}
        validate_plot_outputs(plots_dir, expected=expected)
    print(json.dumps({"plots_dir": str(plots_dir), "entries": entries}, indent=2))


if __name__ == "__main__":
    main()
