"""Paper-aligned CARLTON metrics (CQ, ANCC, WS) vs TF centralized baseline."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT.parent / "carlton-paper-baseline"

try:
    from parse_training_logs import MIN_ACCEPTED_EPISODES, REQUIRED_EPISODES, is_complete_federated_history
except ImportError:
    from experiments.parse_training_logs import MIN_ACCEPTED_EPISODES, REQUIRED_EPISODES, is_complete_federated_history

TF_TRAIN_INFO = BASELINE_ROOT / "train_info.pk"
PYTORCH_TRAIN_INFO = REPO_ROOT / "train_info.pk"

PAPER_METRIC_FIELDS = (
    ("cq_mean", "CQ mean"),
    ("cq_median", "CQ median"),
    ("cq_min", "min CQ"),
    ("ancc", "ANCC"),
    ("anccs", "ANCCS"),
    ("cts", "CTS"),
    ("ses", "SES"),
    ("ws", "WS"),
    ("reward", "Reward (last 20)"),
    ("cc", "Avg channel changes (last 20)"),
)


def _last_mean(values, window=20):
    chunk = [float(v) for v in values[-window:]]
    return float(np.mean(chunk)) if chunk else float("nan")


def _get_vec(train_info, *names):
    for name in names:
        if hasattr(train_info, name):
            values = getattr(train_info, name)
            if values is not None and len(values) > 0:
                return values
    return None


def summarize_train_info(path, label, required=MIN_ACCEPTED_EPISODES, window=20):
    path = Path(path)
    repo_root = path.parent
    for candidate in (repo_root, BASELINE_ROOT, REPO_ROOT):
        candidate = str(candidate)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    with path.open("rb") as f:
        train_info = pickle.load(f)

    rewards = _get_vec(train_info, "average_accumulated_reward_vec")
    if rewards is None or len(rewards) < required:
        raise ValueError(f"{label}: need {required} episodes in {path}, got {len(rewards or [])}")

    sl = slice(-window, None)
    cc = _get_vec(train_info, "average_changed_channels_vec") or []

    metrics = {
        "method": label,
        "source": str(path),
        "episodes": len(rewards),
        "reward": _last_mean(rewards, window),
        "cc": _last_mean(cc, window) if cc else float("nan"),
        "cq_mean": _last_mean(_get_vec(train_info, "cq_mean_vec_train") or [], window),
        "cq_median": _last_mean(_get_vec(train_info, "cq_median") or [], window),
        "cq_min": _last_mean(_get_vec(train_info, "cq_min") or [], window),
        "ancc": _last_mean(_get_vec(train_info, "ancc_vec_train") or [], window),
        "anccs": _last_mean(_get_vec(train_info, "ancc_score_vec_train") or [], window),
        "cts": _last_mean(_get_vec(train_info, "ct_score_vec_train") or [], window),
        "ses": _last_mean(_get_vec(train_info, "se_vec_train") or [], window),
        "ws": _last_mean(_get_vec(train_info, "ws_vec_train") or [], window),
    }
    if np.isnan(metrics["cq_median"]) and not np.isnan(metrics["cq_mean"]):
        metrics["cq_median"] = metrics["cq_mean"]
    if np.isnan(metrics["ancc"]) and not np.isnan(metrics["cc"]):
        metrics["ancc"] = metrics["cc"]
    if np.isnan(metrics["anccs"]) and not np.isnan(metrics["cc"]):
        metrics["anccs"] = max(0.0, 1.0 - metrics["cc"] / 20.0)
    return metrics


def _find_final_weights(run_dir):
    run_dir = Path(run_dir)
    candidates = sorted(run_dir.glob("global_weights_round_*.pkl"))
    if not candidates:
        return None
    return candidates[-1]


def evaluate_fl_weights(weights_path, label, n_eval=20, number_of_channels=10, cache_path=None):
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.is_file():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        data["method"] = label
        return data

    from compare_to_paper import evaluate_global_model

    with Path(weights_path).open("rb") as f:
        global_weights = pickle.load(f)

    rows = evaluate_global_model(global_weights, n_eval=n_eval, number_of_channels=number_of_channels)
    metrics = {
        "method": label,
        "source": str(weights_path),
        "episodes": n_eval,
        "reward": float(np.mean([r["avg_reward"] for r in rows])),
        "cc": float(np.mean([r["avg_channel_changes"] for r in rows])),
        "cq_mean": float(np.mean([r["cq_mean"] for r in rows])),
        "cq_median": float(np.mean([r["cq_median"] for r in rows])),
        "cq_min": float(np.mean([r["cq_min"] for r in rows])),
        "ancc": float(np.mean([r["ancc"] for r in rows])),
        "anccs": float(np.mean([r["ancc_score"] for r in rows])),
        "cts": float(np.mean([r["ct_score"] for r in rows])),
        "ses": float(np.mean([r["se"] for r in rows])),
        "ws": float(np.mean([r["ws"] for r in rows])),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def metrics_from_fl_history(history_path, label, n_eval=20):
    history_path = Path(history_path)
    if not is_complete_federated_history(history_path):
        return None
    weights = _find_final_weights(history_path.parent)
    if weights is None:
        return None
    cache = history_path.parent / f"paper_metrics_eval_n{n_eval}.json"
    return evaluate_fl_weights(weights, label, n_eval=n_eval, cache_path=cache)


def load_tf_baseline_metrics(required=MIN_ACCEPTED_EPISODES):
    return summarize_train_info(TF_TRAIN_INFO, "TF without FL", required=required)


def load_pytorch_baseline_metrics(required=MIN_ACCEPTED_EPISODES, n_eval=20):
    metrics = summarize_train_info(PYTORCH_TRAIN_INFO, "PyTorch without FL", required=required)
    if not np.isnan(metrics.get("ws", float("nan"))):
        return metrics

    wt_dir = REPO_ROOT / "Train_weights_"
    weights = sorted(wt_dir.glob("*.pkl")) if wt_dir.is_dir() else []
    if not weights:
        return metrics

    cache = REPO_ROOT / "experiments" / "results" / "pytorch_centralized_paper_metrics.json"
    eval_metrics = evaluate_fl_weights(
        weights[-1], "PyTorch without FL", n_eval=n_eval, cache_path=cache
    )
    for key in ("ws", "cts", "ses", "cq_median", "cq_min", "ancc", "anccs"):
        if np.isnan(metrics.get(key, float("nan"))) and key in eval_metrics:
            metrics[key] = eval_metrics[key]
    return metrics


def compare_to_tf(rows, baseline):
    compared = []
    for row in rows:
        entry = dict(row)
        entry["vs_tf_pct"] = {}
        for key, _label in PAPER_METRIC_FIELDS:
            base = baseline.get(key)
            val = row.get(key)
            if base is None or val is None or np.isnan(base) or np.isnan(val) or base == 0:
                entry["vs_tf_pct"][key] = None
            else:
                entry["vs_tf_pct"][key] = 100.0 * float(val) / float(base)
        compared.append(entry)
    return compared


def collect_campaign_paper_metrics(campaign_dir, n_eval=20):
    campaign_dir = Path(campaign_dir)
    baseline = None
    try:
        baseline = load_tf_baseline_metrics()
    except (FileNotFoundError, ValueError) as exc:
        baseline_error = str(exc)
    else:
        baseline_error = None

    rows = []
    if baseline is not None:
        rows.append(baseline)
    else:
        rows.append({"method": "TF without FL", "error": baseline_error or "baseline unavailable"})

    try:
        rows.append(load_pytorch_baseline_metrics(n_eval=n_eval))
    except (FileNotFoundError, ValueError) as exc:
        rows.append({"method": "PyTorch without FL", "error": str(exc)})

    best_e, best_mu = 10, 0.01
    manifest = campaign_dir / "campaign_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        best_e = data.get("best_e", best_e)
        best_mu = data.get("best_mu", best_mu)

    mu_tag = str(best_mu).replace(".", "p")
    finals = [
        (f"FedAvg best random (E={best_e})", campaign_dir / "finals" / f"fedavg_E{best_e}_random" / "history.json"),
        (
            f"FedProx best random (E={best_e}, mu={best_mu})",
            campaign_dir / "finals" / f"fedprox_E{best_e}_mu{mu_tag}_random" / "history.json",
        ),
        (f"FedAvg best fixed-6 (E={best_e})", campaign_dir / "finals" / f"fedavg_E{best_e}_fixed6" / "history.json"),
        (
            f"FedProx best fixed-6 (E={best_e}, mu={best_mu})",
            campaign_dir / "finals" / f"fedprox_E{best_e}_mu{mu_tag}_fixed6" / "history.json",
        ),
    ]
    for label, hist in finals:
        if not hist.is_file():
            continue
        try:
            row = metrics_from_fl_history(hist, label, n_eval=n_eval)
            if row is not None:
                rows.append(row)
        except Exception as exc:
            rows.append({"method": label, "error": str(exc)})

    rows = [r for r in rows if r is not None]
    if baseline is not None:
        return compare_to_tf(rows, baseline), baseline
    return rows, None
