"""Build HTML report with plots and paper-comparison metrics table."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from parse_training_logs import (
        REQUIRED_EPISODES,
        MIN_ACCEPTED_EPISODES,
        is_complete_federated_history,
        load_centralized_traces,
        load_federated_history,
    )
except ImportError:
    from experiments.parse_training_logs import (
        REQUIRED_EPISODES,
        MIN_ACCEPTED_EPISODES,
        is_complete_federated_history,
        load_centralized_traces,
        load_federated_history,
    )

try:
    from paper_metrics import PAPER_METRIC_FIELDS, collect_campaign_paper_metrics
except ImportError:
    from experiments.paper_metrics import PAPER_METRIC_FIELDS, collect_campaign_paper_metrics

PAPER_CEILING = 88.0
PAPER_TRAINED_REWARD = 79.0  # qualitative convergence from prior centralized runs


def _mu_tag(mu):
    return str(mu).replace(".", "p")


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


def _plot_traces(traces, title, output_path, smooth_window=20, value_key="rewards", ylabel="Average reward (smoothed)", hline=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, trace in traces.items():
        values = trace.get(value_key) or trace.get("rewards", [])
        if not values:
            continue
        y = smooth(values, window=smooth_window)
        x = _x_for(values, window=smooth_window)
        ax.plot(x, y, label=label, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Episode / Communication round")
    ax.set_ylabel(ylabel)
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", alpha=0.5, label=f"Reference ({hline})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _collect_main_four_traces(campaign_dir):
    best_e, best_mu = _best_from_manifest(campaign_dir)
    try:
        traces = load_centralized_traces(min_points=MIN_ACCEPTED_EPISODES)
    except (FileNotFoundError, ValueError):
        traces = {}

    fedavg_path = _final_history(campaign_dir, f"fedavg_E{best_e}_random")
    fedprox_path = _final_history(campaign_dir, f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_random")

    if fedavg_path is None:
        for path in _find_sweep_histories(campaign_dir, "fedprox_mu0_E*"):
            fedavg_path = path
            break
    if fedprox_path is None:
        sweep_path = Path(campaign_dir) / "sweeps" / f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}" / "history.json"
        if sweep_path.is_file() and is_complete_federated_history(sweep_path):
            fedprox_path = sweep_path

    if fedavg_path is not None:
        traces[f"FedAvg best (E={best_e})"] = load_federated_history(fedavg_path)
    if fedprox_path is not None:
        traces[f"FedProx best (E={best_e}, μ={best_mu})"] = load_federated_history(fedprox_path)
    return traces


def _load_history_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_from_history(path, label=None):
    if not is_complete_federated_history(path):
        return None
    data = _load_history_json(path)
    rewards = [float(v) for v in data.get("round_rewards", [])]
    cc = [float(v) for v in data.get("round_channel_changes", [])]
    if len(rewards) < 20:
        return None
    r20 = float(np.mean(rewards[-20:]))
    cc20 = float(np.mean(cc[-20:])) if len(cc) >= 20 else float("nan")
    cfg = data.get("config", {})
    return {
        "label": label or data.get("run_name", path.parent.name),
        "reward_last_20": r20,
        "cc_last_20": cc20,
        "pct_of_88": 100.0 * r20 / PAPER_CEILING,
        "pct_of_paper_centralized": 100.0 * r20 / PAPER_TRAINED_REWARD,
        "rounds": len(rewards),
        "E": cfg.get("local_train_steps"),
        "mu": cfg.get("fedprox_mu"),
        "fixed_topology": cfg.get("fixed_topology", False),
        "path": str(path),
    }


def _find_sweep_histories(campaign_dir, pattern):
    sweeps = Path(campaign_dir) / "sweeps"
    if not sweeps.is_dir():
        return []
    paths = sorted(p for p in sweeps.glob(f"{pattern}/history.json") if p.is_file())
    return [p for p in paths if is_complete_federated_history(p)]


def _final_history(campaign_dir, run_name):
    path = Path(campaign_dir) / "finals" / run_name / "history.json"
    return path if path.is_file() and is_complete_federated_history(path) else None


def plot_e_sweep(campaign_dir, plots_dir, smooth_window=20):
    traces = {}
    for path in _find_sweep_histories(campaign_dir, "fedprox_E*_mu*"):
        trace = load_federated_history(path)
        e = trace.get("config", {}).get("local_train_steps", "?")
        traces[f"E={e}"] = trace
    if not traces:
        return None
    return _plot_traces(
        traces,
        f"FedProx E sweep (μ={0.01}, persistent replay)",
        plots_dir / "fedprox_e_sweep.png",
        smooth_window=smooth_window,
    )


def plot_mu_sweep(campaign_dir, plots_dir, smooth_window=20):
    traces = {}
    for path in _find_sweep_histories(campaign_dir, "fedprox_mu*_E*"):
        trace = load_federated_history(path)
        mu = trace.get("config", {}).get("fedprox_mu", "?")
        label = "FedAvg (μ=0)" if float(mu) == 0.0 else f"μ={mu}"
        traces[label] = trace
    if not traces:
        return None
    return _plot_traces(
        traces,
        f"FedProx μ sweep (E={1})",
        plots_dir / "fedprox_mu_sweep.png",
        smooth_window=smooth_window,
    )


def _best_from_manifest(campaign_dir):
    manifest_path = Path(campaign_dir) / "campaign_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("best_e", 10), manifest.get("best_mu", 0.01)
    return 10, 0.01


def plot_main_four_methods(campaign_dir, plots_dir, smooth_window=20):
    traces = _collect_main_four_traces(campaign_dir)
    if len(traces) < 2:
        return None
    return _plot_traces(
        traces,
        "Main comparison: 4 methods (reward drift)",
        plots_dir / "main_four_methods.png",
        smooth_window=smooth_window,
        value_key="rewards",
        ylabel="Average reward (smoothed)",
        hline=PAPER_CEILING,
    )


def plot_main_four_methods_cc(campaign_dir, plots_dir, smooth_window=20):
    traces = _collect_main_four_traces(campaign_dir)
    if len(traces) < 2:
        return None
    return _plot_traces(
        traces,
        "Main comparison: 4 methods (channel changes)",
        plots_dir / "main_four_methods_cc.png",
        smooth_window=smooth_window,
        value_key="channel_changes",
        ylabel="Avg channel changes per round (smoothed)",
        hline=None,
    )


def plot_random_vs_fixed(campaign_dir, plots_dir, smooth_window=20):
    best_e, best_mu = _best_from_manifest(campaign_dir)
    traces = {}
    mapping = [
        (f"FedAvg random (E={best_e})", _final_history(campaign_dir, f"fedavg_E{best_e}_random")),
        (f"FedAvg fixed-6 (E={best_e})", _final_history(campaign_dir, f"fedavg_E{best_e}_fixed6")),
        (f"FedProx random (E={best_e}, μ={best_mu})", _final_history(campaign_dir, f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_random")),
        (f"FedProx fixed-6 (E={best_e}, μ={best_mu})", _final_history(campaign_dir, f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_fixed6")),
    ]
    for label, path in mapping:
        if path is not None:
            traces[label] = load_federated_history(path)
    if len(traces) < 2:
        return None
    return _plot_traces(
        traces,
        "Best config: random topology vs fixed 6-network map",
        plots_dir / "random_vs_fixed.png",
        smooth_window=smooth_window,
    )


def collect_metrics_table(campaign_dir):
    rows = []

    try:
        traces = load_centralized_traces(min_points=MIN_ACCEPTED_EPISODES)
        for label, log_key in (
            ("TF centralized (no FL)", "TF without FL"),
            ("PyTorch centralized (no FL)", "PyTorch without FL"),
        ):
            if log_key not in traces:
                continue
            rewards = traces[log_key]["rewards"]
            if len(rewards) < MIN_ACCEPTED_EPISODES:
                continue
            r20 = float(np.mean(rewards[-20:]))
            cc20 = None
            try:
                from paper_metrics import summarize_train_info
                from parse_training_logs import PYTORCH_TRAIN_INFO, TF_TRAIN_INFO
            except ImportError:
                from experiments.paper_metrics import summarize_train_info
                from experiments.parse_training_logs import PYTORCH_TRAIN_INFO, TF_TRAIN_INFO
            pk = TF_TRAIN_INFO if log_key == "TF without FL" else PYTORCH_TRAIN_INFO
            try:
                cc20 = summarize_train_info(pk, label)["cc"]
            except (FileNotFoundError, ValueError):
                pass
            rows.append(
                {
                    "method": label,
                    "reward_last_20": r20,
                    "cc_last_20": cc20,
                    "pct_of_88": 100.0 * r20 / PAPER_CEILING,
                    "pct_vs_paper_train": 100.0 * r20 / PAPER_TRAINED_REWARD,
                    "E": "—",
                    "mu": "—",
                    "env": "random",
                }
            )
    except (FileNotFoundError, ValueError):
        pass

    best_e, best_mu = _best_from_manifest(campaign_dir)
    candidates = [
        ("FedAvg best random", _final_history(campaign_dir, f"fedavg_E{best_e}_random")),
        ("FedAvg best fixed-6", _final_history(campaign_dir, f"fedavg_E{best_e}_fixed6")),
        ("FedProx best random", _final_history(campaign_dir, f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_random")),
        ("FedProx best fixed-6", _final_history(campaign_dir, f"fedprox_E{best_e}_mu{_mu_tag(best_mu)}_fixed6")),
    ]
    for label, path in candidates:
        if path is None:
            continue
        m = _metrics_from_history(path, label)
        if m:
            rows.append(
                {
                    "method": label,
                    "reward_last_20": m["reward_last_20"],
                    "cc_last_20": m["cc_last_20"],
                    "pct_of_88": m["pct_of_88"],
                    "pct_vs_paper_train": m["pct_of_paper_centralized"],
                    "E": m["E"],
                    "mu": m["mu"],
                    "env": "fixed" if m.get("fixed_topology") else "random",
                }
            )

    for path in _find_sweep_histories(campaign_dir, "fedprox_E*_mu*"):
        e = _load_history_json(path).get("config", {}).get("local_train_steps")
        m = _metrics_from_history(path, f"FedProx E={e} (μ=0.01)")
        if m:
            rows.append(
                {
                    "method": m["label"],
                    "reward_last_20": m["reward_last_20"],
                    "cc_last_20": m["cc_last_20"],
                    "pct_of_88": m["pct_of_88"],
                    "pct_vs_paper_train": m["pct_of_paper_centralized"],
                    "E": m["E"],
                    "mu": m["mu"],
                    "env": "random",
                }
            )

    for path in _find_sweep_histories(campaign_dir, "fedprox_mu*_E*"):
        mu = _load_history_json(path).get("config", {}).get("fedprox_mu")
        m = _metrics_from_history(path, f"FedProx μ={mu} (E=1)")
        if m:
            rows.append(
                {
                    "method": m["label"],
                    "reward_last_20": m["reward_last_20"],
                    "cc_last_20": m["cc_last_20"],
                    "pct_of_88": m["pct_of_88"],
                    "pct_vs_paper_train": m["pct_of_paper_centralized"],
                    "E": m["E"],
                    "mu": m["mu"],
                    "env": "random",
                }
            )

    return rows


def _img_data_uri(path):
    path = Path(path)
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _html_paper_metrics_table(rows):
    if not rows:
        return "<p>Paper metrics not available yet (waiting for TF/PyTorch baselines with &ge; 900 episodes).</p>"

    metric_keys = [k for k, _ in PAPER_METRIC_FIELDS]
    headers = ["Method"] + [label for _, label in PAPER_METRIC_FIELDS] + ["WS vs TF"]
    lines = ["<table><thead><tr>"]
    for h in headers:
        lines.append(f"<th>{h}</th>")
    lines.append("</tr></thead><tbody>")

    for row in rows:
        if row.get("error"):
            lines.append(f"<tr><td>{row['method']}</td><td colspan='{len(headers)-1}'><em>{row['error']}</em></td></tr>")
            continue
        lines.append("<tr>")
        lines.append(f"<td>{row['method']}</td>")
        for key in metric_keys:
            val = row.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                lines.append("<td>—</td>")
            else:
                pct = row.get("vs_tf_pct", {}).get(key)
                if pct is not None and row["method"] != "TF without FL":
                    lines.append(f"<td>{float(val):.3f}<br><small>({pct:.0f}% of TF)</small></td>")
                else:
                    lines.append(f"<td>{float(val):.3f}</td>")
        ws_pct = row.get("vs_tf_pct", {}).get("ws")
        if row["method"] == "TF without FL":
            lines.append("<td>reference</td>")
        elif ws_pct is None:
            lines.append("<td>—</td>")
        else:
            lines.append(f"<td>{ws_pct:.0f}%</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def plot_paper_metrics_vs_tf(rows, output_path):
    usable = [r for r in rows if not r.get("error") and r.get("ws") is not None]
    if len(usable) < 2:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [r["method"] for r in usable]
    ws_vals = [float(r["ws"]) for r in usable]
    cq_vals = [float(r["cq_mean"]) for r in usable]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, ws_vals, width, label="WS")
    ax.bar(x + width / 2, cq_vals, width, label="CQ mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Paper metrics vs TF without FL (eval / last-20 training window)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _html_table(rows):
    if not rows:
        return "<p>No metrics available yet.</p>"
    headers = ["Method", "Reward (last 20)", "CC (last 20)", "% of 88", "% vs paper train (~79)", "E", "mu", "Env"]
    lines = ["<table><thead><tr>"]
    for h in headers:
        lines.append(f"<th>{h}</th>")
    lines.append("</tr></thead><tbody>")
    for row in rows:
        cc = row["cc_last_20"]
        cc_str = f"{cc:.2f}" if cc is not None and not np.isnan(cc) else "—"
        lines.append("<tr>")
        lines.append(f"<td>{row['method']}</td>")
        lines.append(f"<td>{row['reward_last_20']:.2f}</td>")
        lines.append(f"<td>{cc_str}</td>")
        lines.append(f"<td>{row['pct_of_88']:.1f}%</td>")
        lines.append(f"<td>{row['pct_vs_paper_train']:.1f}%</td>")
        lines.append(f"<td>{row['E']}</td>")
        lines.append(f"<td>{row['mu']}</td>")
        lines.append(f"<td>{row['env']}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _rel_from_docs(path):
    path = Path(path).resolve()
    docs = (REPO_ROOT / "docs").resolve()
    return Path(os.path.relpath(path, docs)).as_posix()


def generate_theory_campaign_section(
    campaign_dir,
    rows,
    paper_rows,
    plot_paths,
    best_e,
    best_mu,
    tf_status,
    pytorch_status,
):
    """HTML fragment for docs/federated_learning_theory.html (section 15.7)."""
    campaign_dir = Path(campaign_dir)
    plots_rel = campaign_dir.name

    def plot_img(key, alt):
        path = plot_paths.get(key)
        if not isinstance(path, Path) or not path.is_file():
            return ""
        src = _rel_from_docs(path)
        return f'<div class="diagram campaign-plot"><img src="{src}" alt="{alt}"/><p class="campaign-meta">{alt}</p></div>'

    plot_blocks = []
    for title, key in (
        ("Main comparison — 4 methods (reward)", "main_four"),
        ("Main comparison — 4 methods (channel changes)", "main_four_cc"),
        ("Paper metrics vs TF without FL", "paper_metrics"),
        ("FedProx E sweep (μ=0.01)", "e_sweep"),
        ("FedProx μ sweep (E=1)", "mu_sweep"),
        ("Best config: random vs fixed topology", "random_vs_fixed"),
    ):
        block = plot_img(key, title)
        if block:
            plot_blocks.append(f"<h4>{title}</h4>\n{block}")

    return f"""  <h3>15.7 Campaign results — FedProx / FedAvg</h3>
  <p class="campaign-meta"><strong>Campaign:</strong> {campaign_dir.name} &nbsp;|&nbsp;
     <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp;
     <strong>Best E:</strong> {best_e} &nbsp;|&nbsp; <strong>Best μ:</strong> {best_mu}</p>

  <div class="callout def">
    <strong>Paper reference</strong>
    Perfect-game reward ceiling ≈ 88; centralized training convergence ≈ 79
    (<code>paper_reference/CARLTON_paper_results.md</code>).
    Baseline for paper metrics: TF without FL (CTDE) — <em>{tf_status}</em>.
    PyTorch without FL — <em>{pytorch_status}</em>.
    Only runs with ≥ {MIN_ACCEPTED_EPISODES} episodes/rounds are included.
  </div>

  <h4>Reward metrics (last-20 mean)</h4>
  {_html_table(rows)}

  <h4>Paper metrics vs TF without FL</h4>
  <p>CQ = Channel Quality (QV at final scenario time). ANCCS, CTS, SES, and WS follow Eq. 18–22.
     FL rows use post-training evaluation on final global weights (20 games). Centralized rows use the last-20-episode training mean.</p>
  {_html_paper_metrics_table(paper_rows)}

  <h4>Learning curves and sweeps</h4>
  {"\n\n".join(plot_blocks) if plot_blocks else "<p><em>Plots not available.</em></p>"}

  <div class="callout good">
    <strong>How to refresh this section</strong>
    <code>py -3 experiments/build_html_report.py --campaign-dir experiments/results/{plots_rel}</code>
  </div>"""


THEORY_DOC = REPO_ROOT / "docs" / "federated_learning_theory.html"
CAMPAIGN_MARKER_START = "<!-- CAMPAIGN_RESULTS_START -->"
CAMPAIGN_MARKER_END = "<!-- CAMPAIGN_RESULTS_END -->"


def inject_campaign_into_theory(section_html):
    if not THEORY_DOC.is_file():
        return False
    text = THEORY_DOC.read_text(encoding="utf-8")
    start = text.find(CAMPAIGN_MARKER_START)
    end = text.find(CAMPAIGN_MARKER_END)
    if start == -1 or end == -1:
        return False
    new_text = (
        text[: start + len(CAMPAIGN_MARKER_START)]
        + "\n"
        + section_html
        + "\n  "
        + text[end:]
    )
    THEORY_DOC.write_text(new_text, encoding="utf-8")
    return True


def build_report(campaign_dir, output_path=None, inject_theory=True):
    campaign_dir = Path(campaign_dir)
    plots_dir = campaign_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}
    for name, fn in (
        ("e_sweep", plot_e_sweep),
        ("mu_sweep", plot_mu_sweep),
        ("main_four", plot_main_four_methods),
        ("main_four_cc", plot_main_four_methods_cc),
        ("random_vs_fixed", plot_random_vs_fixed),
    ):
        try:
            path = fn(campaign_dir, plots_dir)
            if path:
                plot_paths[name] = path
        except Exception as exc:
            plot_paths[name] = f"error: {exc}"

    rows = collect_metrics_table(campaign_dir)
    metrics_path = campaign_dir / "metrics_table.json"
    metrics_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    paper_rows = []
    paper_baseline = None
    try:
        paper_rows, paper_baseline = collect_campaign_paper_metrics(campaign_dir, n_eval=20)
        (campaign_dir / "paper_metrics_table.json").write_text(
            json.dumps({"baseline": paper_baseline, "rows": paper_rows}, indent=2),
            encoding="utf-8",
        )
        if paper_baseline is not None:
            paper_plot = plot_paper_metrics_vs_tf(paper_rows, plots_dir / "paper_metrics_vs_tf.png")
            if paper_plot:
                plot_paths["paper_metrics"] = paper_plot
    except Exception as exc:
        paper_rows = [{"method": "Paper metrics", "error": str(exc)}]

    manifest = {}
    mp = campaign_dir / "campaign_manifest.json"
    if mp.is_file():
        manifest = json.loads(mp.read_text(encoding="utf-8"))

    output_path = Path(output_path) if output_path else campaign_dir / "report.html"
    best_e = manifest.get("best_e", "?")
    best_mu = manifest.get("best_mu", "?")

    tf_status = "not available"
    pytorch_status = "not available"
    try:
        from parse_training_logs import TF_TRAIN_INFO, PYTORCH_TRAIN_INFO, load_train_info_rewards
    except ImportError:
        from experiments.parse_training_logs import TF_TRAIN_INFO, PYTORCH_TRAIN_INFO, load_train_info_rewards
    try:
        tf_trace = load_train_info_rewards(TF_TRAIN_INFO, "TF without FL")
        tf_status = f"{len(tf_trace['rewards'])} episodes"
    except (FileNotFoundError, ValueError):
        pass
    try:
        py_trace = load_train_info_rewards(PYTORCH_TRAIN_INFO, "PyTorch without FL")
        pytorch_status = f"{len(py_trace['rewards'])} episodes"
    except (FileNotFoundError, ValueError):
        pass

    sections = []
    for title, key in (
        ("Main comparison — 4 methods (reward)", "main_four"),
        ("Main comparison — 4 methods (channel changes)", "main_four_cc"),
        ("Paper metrics vs TF without FL", "paper_metrics"),
        ("FedProx E sweep", "e_sweep"),
        ("FedProx mu sweep", "mu_sweep"),
        ("Best config: random vs fixed topology", "random_vs_fixed"),
    ):
        path = plot_paths.get(key)
        if isinstance(path, Path) and path.is_file():
            sections.append(f'<h2>{title}</h2><img src="{_img_data_uri(path)}" alt="{title}" style="max-width:100%"/>')
        elif path:
            sections.append(f"<h2>{title}</h2><p><em>Not available: {path}</em></p>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>FedProx / FedAvg Campaign Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: center; }}
    th {{ background: #eee; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .meta {{ background: #fff; padding: 1rem; border: 1px solid #ddd; margin-bottom: 1.5rem; }}
    img {{ background: #fff; border: 1px solid #ddd; padding: 0.5rem; max-width: 100%; }}
    small {{ color: #555; }}
  </style>
</head>
<body>
  <h1>FedProx / FedAvg Campaign Report — CARLTON FRL</h1>
  <div class="meta">
    <p><strong>Campaign:</strong> {campaign_dir.name}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    <p><strong>Best E:</strong> {best_e} &nbsp;|&nbsp; <strong>Best mu:</strong> {best_mu}</p>
    <p><strong>Paper reference:</strong> perfect-game ceiling ~88; centralized training convergence ~79
      (<code>paper_reference/CARLTON_paper_results.md</code>)</p>
    <p><strong>Baseline for paper metrics:</strong> TF without FL (CTDE) — <em>{tf_status}</em></p>
    <p><strong>PyTorch without FL:</strong> <em>{pytorch_status}</em></p>
  </div>

  <h2>Reward metrics (runs with &ge; {MIN_ACCEPTED_EPISODES} episodes/rounds)</h2>
  {_html_table(rows)}

  <h2>Paper metrics vs TF without FL</h2>
  <p>CQ = Channel Quality (QV at final scenario time). ANCCS, CTS, SES, and WS follow Eq. 18–22 in the paper.
     FL rows use post-training evaluation on final global weights (20 games). Centralized rows use the last-20-episode training mean.</p>
  {_html_paper_metrics_table(paper_rows)}

  {"".join(sections)}

  <h2>Notes</h2>
  <ul>
    <li>Reward = mean accumulated reward per round/episode (smoothed with window 20 in plots).</li>
    <li>CC = mean channel changes per round (lower is more stable).</li>
    <li>FedAvg is FedProx with mu=0.</li>
    <li>Only runs with at least {MIN_ACCEPTED_EPISODES} episodes/rounds are included.</li>
    <li>WS = 0.4&middot;CQ + 0.4&middot;CTS + 0.1&middot;ANCCS + 0.1&middot;SES (paper Eq. 22).</li>
  </ul>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")

    if inject_theory:
        section = generate_theory_campaign_section(
            campaign_dir,
            rows,
            paper_rows,
            plot_paths,
            best_e,
            best_mu,
            tf_status,
            pytorch_status,
        )
        inject_campaign_into_theory(section)

    return output_path


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--no-inject-theory", action="store_true")
    args = parser.parse_args(argv)
    path = build_report(
        args.campaign_dir,
        output_path=args.output,
        inject_theory=not args.no_inject_theory,
    )
    result = {"report": str(path)}
    if not args.no_inject_theory:
        result["theory_doc"] = str(THEORY_DOC)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
