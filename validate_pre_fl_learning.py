"""Quick pre-FL validation: centralized CARLTON (paper CTDE) learning check.

Runs a short episode loop with global replay + train_model.
Stops early if no learning signal is detected by mid-run.
"""

import json
import os
import shutil
from datetime import datetime

import numpy as np

from BuildingBlocks.TrainBlock import train_model
from BuildingBlocks.Worker import worker
from SimulationEnvironments.Pythonic_Environment import python_env
from Utils.RandomLocationOfNetworks import set_random_location_of_networks


def _mean_window(values, start, end):
    chunk = values[start:end]
    return float(np.mean(chunk)) if chunk else float("nan")


def assess_learning(rewards, channel_changes, costs_by_ep):
    n = len(rewards)
    if n < 20:
        return {"status": "insufficient_data", "message": "Need at least 20 episodes"}

    r_early = _mean_window(rewards, 0, min(10, n))
    r_late = _mean_window(rewards, max(0, n - 10), n)
    cc_early = _mean_window(channel_changes, 0, min(10, n))
    cc_late = _mean_window(channel_changes, max(0, n - 10), n)

    reward_improved = r_late > r_early * 1.03
    mobility_improved = cc_late < cc_early * 0.90
    loss_trend_ok = False
    if len(costs_by_ep) >= 10:
        first_costs = [c for c in costs_by_ep[:5] if c is not None]
        last_costs = [c for c in costs_by_ep[-5:] if c is not None]
        if first_costs and last_costs:
            loss_trend_ok = np.mean(last_costs) < np.mean(first_costs)

    signals = sum([reward_improved, mobility_improved, loss_trend_ok])
    passed = signals >= 2

    return {
        "status": "pass" if passed else "fail",
        "reward_early": r_early,
        "reward_late": r_late,
        "reward_improved": reward_improved,
        "channel_changes_early": cc_early,
        "channel_changes_late": cc_late,
        "mobility_improved": mobility_improved,
        "loss_trend_ok": loss_trend_ok,
        "learning_signals": signals,
        "pct_of_paper_ceiling_88": 100.0 * r_late / 88.0,
    }


def run_pre_fl_validation(
    max_episodes=40,
    early_check_at=20,
    train_iterations=40,
    i_d_folder="pre_fl",
    number_of_channels=10,
):
    base_dir = os.getcwd()
    rb_dir = os.path.join(base_dir, f"Global_RB_Storage_{i_d_folder}")
    wt_dir = os.path.join(base_dir, f"Train_weights_{i_d_folder}")
    for d in (rb_dir, wt_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)

    rewards = []
    channel_changes = []
    costs_by_ep = []
    epsilon = 0.5
    epsilon_end = 0.01
    eps_decay = (epsilon - epsilon_end) / max(1, max_episodes // 2)

    print(f"=== Pre-FL centralized CARLTON validation ({max_episodes} episodes) ===", flush=True)

    for ep in range(max_episodes):
        n_nets = np.random.randint(2, 8)
        users, centers = set_random_location_of_networks(n_nets)
        scenario = python_env(
            number_of_nets=n_nets,
            number_of_users_in_each_net=users,
            net_center_location_and_std=centers,
            possible_channels=number_of_channels,
            add_noise=False,
            training=True,
        )

        avg_reward, avg_cc, _, _ = worker(
            address_scen="",
            scenario=scenario,
            address_algo=base_dir,
            training=True,
            epsilon=epsilon,
            local_train_steps=0,
            save_to_global_rb=True,
            i_d_folder=i_d_folder,
            batch_size=32,
            replay_memory_size=100000,
            verbose=False,
        )

        costs = train_model(
            trainig_iterations=train_iterations,
            batch_size=32,
            action_space=number_of_channels,
            i_d_folder=i_d_folder,
            base_dir=base_dir,
            verbose=False,
        )
        mean_cost = float(np.mean(costs)) if costs else None

        rewards.append(float(avg_reward))
        channel_changes.append(float(avg_cc))
        costs_by_ep.append(mean_cost)

        if ep < max_episodes // 2:
            epsilon = max(epsilon_end, epsilon - eps_decay)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"Ep {ep + 1}/{max_episodes} | reward={avg_reward:.2f} | "
                f"cc={avg_cc:.2f} | loss={mean_cost} | eps={epsilon:.3f}",
                flush=True,
            )

        if ep + 1 == early_check_at:
            mid = assess_learning(rewards, channel_changes, costs_by_ep)
            print(f"\n--- Early check @ episode {early_check_at}: {mid['status'].upper()} ---", flush=True)
            print(json.dumps(mid, indent=2), flush=True)
            if mid["status"] == "fail":
                print("\nSTOP: No clear learning signal. Debug before full 1000-episode run.", flush=True)
                return {
                    "episodes_run": ep + 1,
                    "early_stop": True,
                    "final": mid,
                    "rewards": rewards,
                    "channel_changes": channel_changes,
                }

    final = assess_learning(rewards, channel_changes, costs_by_ep)
    print(f"\n=== Final assessment: {final['status'].upper()} ===", flush=True)
    print(json.dumps(final, indent=2), flush=True)

    report_path = os.path.join("paper_reference", "pre_fl_validation_report.json")
    os.makedirs("paper_reference", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "episodes": max_episodes,
                "early_stop": False,
                "final": final,
                "rewards": rewards,
                "channel_changes": channel_changes,
                "costs": costs_by_ep,
            },
            f,
            indent=2,
        )
    print(f"Report saved: {report_path}", flush=True)
    return final


if __name__ == "__main__":
    run_pre_fl_validation()
