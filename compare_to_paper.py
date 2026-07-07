"""Evaluate FRL run against paper metrics and write comparison report."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from main_v3 import run_federated_training
from BuildingBlocks.Worker import worker
from Utils.RandomLocationOfNetworks import set_random_location_of_networks
from Utils.ScenarioExamination import get_game_performamce
from Utils.save_to_df_csv import wrrape_game_history_do_df
from SimulationEnvironments.Pythonic_Environment import python_env


PAPER = {
    "training_episodes": 1000,
    "eval_games": 420,
    "networks_train_max": 7,
    "networks_eval_range": (2, 15),
    "perfect_game_reward": 88,
    "rho": 0.7,
    "min_cq_at_convergence": 0.95,
    "ses_typical": 0.8,
    "margin_vs_ra_pct": 45,
    "margin_vs_jar_pct": 20,
    "gap_vs_graph_coloring_pct": 2.5,
    "ws_weights": (0.4, 0.1, 0.4, 0.1),  # CQ, ANCCS, CTS, SES
}


def evaluate_global_model(global_weights, n_eval=10, number_of_channels=10):
    metrics = []
    for _ in range(n_eval):
        n_nets = np.random.randint(2, 8)
        users, centers = set_random_location_of_networks(n_nets)
        env = python_env(
            number_of_nets=n_nets,
            number_of_users_in_each_net=users,
            net_center_location_and_std=centers,
            possible_channels=number_of_channels,
            add_noise=False,
            training=True,
        )
        avg_reward, avg_cc, _, game_history = worker(
            address_scen="",
            scenario=env,
            address_algo="",
            training=False,
            epsilon=0.0,
            global_weights=global_weights,
            local_train_steps=0,
            verbose=False,
        )
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
            cq_above90,
            cq_below90,
            se,
            ancc_score,
            ct_score,
            ws,
            _,
            _,
        ) = perf
        metrics.append(
            {
                "networks": n_nets,
                "avg_reward": float(avg_reward),
                "avg_channel_changes": float(avg_cc),
                "cq_mean": float(cq_mean),
                "cq_min": float(cq_min),
                "cq_median": float(cq_median),
                "ancc": float(ancc),
                "ancc_score": float(ancc_score),
                "ct_score": float(ct_score),
                "se": float(se),
                "ws": float(ws),
            }
        )
    return metrics


def summarize_training_history(history):
    rewards = np.array(history["round_rewards"], dtype=float)
    cc = np.array(history["round_channel_changes"], dtype=float)
    half = max(1, len(rewards) // 2)
    return {
        "rounds": len(rewards),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_first_half_mean": float(rewards[:half].mean()),
        "reward_second_half_mean": float(rewards[half:].mean()),
        "channel_changes_mean": float(cc.mean()),
        "channel_changes_first_half_mean": float(cc[:half].mean()),
        "channel_changes_second_half_mean": float(cc[half:].mean()),
        "pct_of_perfect_game_reward": float(100.0 * rewards.mean() / PAPER["perfect_game_reward"]),
    }


def write_report(train_summary, eval_summary, eval_rows, out_path):
    lines = [
        "# FRL Run vs CARLTON Paper — Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. Setup differences (important)",
        "",
        "| Aspect | Paper CARLTON | This FRL run |",
        "|--------|---------------|--------------|",
        f"| Training length | {PAPER['training_episodes']} episodes | {train_summary['rounds']} federated rounds |",
        "| Aggregation | Global replay + centralized NN update | FedAvg over local `state_dict` |",
        f"| Reward ρ | {PAPER['rho']} | 0.7 (Coordinator.RHO) |",
        f"| Eval games | {PAPER['eval_games']} | {len(eval_rows)} post-train scenarios |",
        "",
        "## 2. Training run (logged by main_v3)",
        "",
        "| Metric | Our run | Paper reference |",
        "|--------|---------|-----------------|",
        f"| Mean accumulated reward / round | {train_summary['reward_mean']:.2f} | Perfect game ≈ {PAPER['perfect_game_reward']} (Fig. 3) |",
        f"| Reward range | {train_summary['reward_min']:.2f} – {train_summary['reward_max']:.2f} | — |",
        f"| Mean reward (rounds 1–{train_summary['rounds']//2}) | {train_summary['reward_first_half_mean']:.2f} | — |",
        f"| Mean reward (rounds {train_summary['rounds']//2+1}–{train_summary['rounds']}) | {train_summary['reward_second_half_mean']:.2f} | — |",
        f"| % of paper perfect-game ceiling | {train_summary['pct_of_perfect_game_reward']:.1f}% | 100% at convergence |",
        f"| Mean channel changes / agent | {train_summary['channel_changes_mean']:.2f} | Lower ANCC is better; ANCCS = 1−ANCC/T |",
        f"| Channel changes (1st half) | {train_summary['channel_changes_first_half_mean']:.2f} | — |",
        f"| Channel changes (2nd half) | {train_summary['channel_changes_second_half_mean']:.2f} | Paper: ANCCS improves with training |",
        "",
        "## 3. Post-train evaluation (paper-aligned metrics)",
        "",
        "Computed via `get_game_performamce()` (WS uses repo weights 0.4 CQ + 0.4 CT + 0.1 ANCC + 0.1 SE).",
        "",
        "| Metric | Our eval (mean) | Paper target / trend | Assessment |",
        "|--------|-----------------|----------------------|------------|",
        f"| WS | {eval_summary['ws_mean']:.3f} | Best among distributed baselines (Fig. 12) | Qualitative only — no numeric paper WS in text |",
        f"| CQ_mean | {eval_summary['cq_mean']:.3f} | Improves to high values at convergence (Fig. 4) | {'OK' if eval_summary['cq_mean'] >= 0.7 else 'Below typical converged CQ'} |",
        f"| min_CQ | {eval_summary['cq_min']:.3f} | > 0.95 for some agents at convergence | {'Near paper' if eval_summary['cq_min'] >= 0.85 else 'Below paper (>0.95)'} |",
        f"| ANCCS | {eval_summary['ancc_score_mean']:.3f} | Improves during training (Eq. 19) | {'Improving' if train_summary['channel_changes_second_half_mean'] < train_summary['channel_changes_first_half_mean'] else 'Mixed'} |",
        f"| CTS | {eval_summary['ct_score_mean']:.3f} | Improves (Eq. 20); lower CTS vs JAR without φ | — |",
        f"| SES (SE) | {eval_summary['se_mean']:.3f} | ~0.8 at convergence (Sec. IV-A) | {'Near paper' if eval_summary['se_mean'] >= 0.65 else 'Below ~0.8'} |",
        f"| E[(CQ+min_CQ)/2] | {eval_summary['cq_minmax_combo']:.3f} | Primary comparison metric (Figs 7, 13, 15) | — |",
        "",
        "## 4. Headline paper claims (not directly reproduced here)",
        "",
        "| Claim | Paper | This run |",
        "|-------|-------|----------|",
        f"| vs Random Agent | ~+{PAPER['margin_vs_ra_pct']}% | **Not tested** (no RA baseline run) |",
        f"| vs JAR | ~+{PAPER['margin_vs_jar_pct']}% | **Not tested** (no JAR baseline run) |",
        f"| vs graph coloring | ~−{PAPER['gap_vs_graph_coloring_pct']}% gap | **Not tested** |",
        "",
        "## 5. Per-scenario eval detail",
        "",
        "| N nets | Reward | CQ_mean | min_CQ | WS | ANCCS | CTS | SE |",
        "|--------|--------|---------|--------|-----|-------|-----|-----|",
    ]
    for row in eval_rows:
        lines.append(
            f"| {row['networks']} | {row['avg_reward']:.1f} | {row['cq_mean']:.3f} | "
            f"{row['cq_min']:.3f} | {row['ws']:.3f} | {row['ancc_score']:.3f} | "
            f"{row['ct_score']:.3f} | {row['se']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 6. Summary verdict",
            "",
        ]
    )

    verdicts = []
    if train_summary["channel_changes_second_half_mean"] < train_summary["channel_changes_first_half_mean"]:
        verdicts.append("- **Spectrum mobility:** Second-half channel changes lower than first half — consistent with paper trend (fewer switches as policy stabilizes).")
    else:
        verdicts.append("- **Spectrum mobility:** No clear decrease in channel changes across training halves.")

    if train_summary["pct_of_perfect_game_reward"] < 80:
        verdicts.append(
            f"- **Reward scale:** Mean reward is {train_summary['pct_of_perfect_game_reward']:.0f}% of paper perfect-game ceiling (88) — expected with only {train_summary['rounds']} rounds vs paper 1000 episodes."
        )

    if eval_summary["cq_min"] < 0.95:
        verdicts.append("- **min_CQ:** Below paper's reported >0.95 at full convergence; more training/eval games needed.")

    verdicts.append("- **Baselines:** To match paper Section IV-B, run RA, JAR, and graph-coloring on the same 420-game protocol.")
    verdicts.append("- **Protocol:** Paper uses CTDE+GRM; this repo uses FRL+FedAvg — not apples-to-apples without aligning training length and ρ=0.7.")

    lines.extend(verdicts)
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {"train_summary": train_summary, "eval_summary": eval_summary, "eval_rows": eval_rows},
            f,
            indent=2,
        )


def main():
    print("Training (20 federated rounds)...")
    history, global_weights = run_federated_training(
        communication_rounds=20,
        local_train_steps=10,
    )
    train_summary = summarize_training_history(history)

    print("Evaluating (10 scenarios, greedy policy)...")
    eval_rows = evaluate_global_model(global_weights, n_eval=10)
    df = pd.DataFrame(eval_rows)

    eval_summary = {
        "ws_mean": float(df["ws"].mean()),
        "cq_mean": float(df["cq_mean"].mean()),
        "cq_min": float(df["cq_min"].mean()),
        "cq_median": float(df["cq_median"].mean()),
        "ancc_score_mean": float(df["ancc_score"].mean()),
        "ct_score_mean": float(df["ct_score"].mean()),
        "se_mean": float(df["se"].mean()),
        "cq_minmax_combo": float(((df["cq_mean"] + df["cq_min"]) / 2).mean()),
    }

    out = __import__("pathlib").Path("paper_reference/comparison_frl_vs_paper.md")
    write_report(train_summary, eval_summary, eval_rows, out)
    print(f"Report written to {out}")
    print(f"Train reward mean: {train_summary['reward_mean']:.2f} ({train_summary['pct_of_perfect_game_reward']:.1f}% of paper ceiling 88)")
    print(f"Eval WS mean: {eval_summary['ws_mean']:.3f}, CQ_mean: {eval_summary['cq_mean']:.3f}, min_CQ: {eval_summary['cq_min']:.3f}")


if __name__ == "__main__":
    main()
