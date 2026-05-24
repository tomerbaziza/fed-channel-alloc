"""Sanity check for federated CARLTON loop.

Runs exactly 2 federated rounds with small local training and verifies:
Step -> Local Train -> Weight Upload -> FedAvg -> Broadcast path executes.
"""

from main_v3 import run_federated_training


def main():
    history, global_weights = run_federated_training(
        communication_rounds=2,
        local_train_steps=2,
        history_length=1,
        number_of_channels=10,
        min_nets=2,
        max_nets=3,
        epsilon_start=0.2,
        epsilon_end=0.05,
        learning_rate=0.00025,
    )

    assert len(history["round_rewards"]) == 2, "Expected exactly 2 federated rounds."
    assert len(history["round_num_agents"]) == 2, "Expected agent-count trace per round."
    assert isinstance(global_weights, dict) and len(global_weights) > 0, "FedAvg did not return valid global weights."
    print("Federated sanity check passed.")


if __name__ == "__main__":
    main()
