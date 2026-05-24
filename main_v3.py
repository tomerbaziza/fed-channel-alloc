"""Federated CARLTON training loop (PyTorch FRL refactor)."""

import numpy as np
from BuildingBlocks.Worker import worker
from BuildingBlocks.TrainBlock import federated_averaging
from Utils.RandomLocationOfNetworks import set_random_location_of_networks
from SimulationEnvironments.Pythonic_Environment import python_env


def run_federated_training(
    communication_rounds=20,
    local_train_steps=10,
    history_length=1,
    number_of_channels=10,
    min_nets=2,
    max_nets=7,
    epsilon_start=0.5,
    epsilon_end=0.01,
    learning_rate=0.00025,
):
    """Run FRL rounds: broadcast -> local train -> upload -> FedAvg."""
    global_weights = None
    epsilon = epsilon_start

    history = {
        "round_rewards": [],
        "round_channel_changes": [],
        "round_num_agents": [],
    }

    for round_idx in range(communication_rounds):
        number_of_nets = np.random.randint(min_nets, max_nets + 1)
        users_per_net, net_centers = set_random_location_of_networks(number_of_nets)
        scenario = python_env(
            number_of_nets=number_of_nets,
            number_of_users_in_each_net=users_per_net,
            net_center_location_and_std=net_centers,
            possible_channels=number_of_channels,
            add_noise=False,
            training=True,
        )

        avg_reward, avg_channel_changes, agents, _ = worker(
            address_scen="",
            scenario=scenario,
            address_algo="",
            training=True,
            history_length=history_length,
            learning_rate=learning_rate,
            epsilon=epsilon,
            global_weights=global_weights,
            local_train_steps=local_train_steps,
            verbose=False,
        )

        local_weights = [agent.state_dict() for agent in agents.values()]
        global_weights = federated_averaging(local_weights)

        history["round_rewards"].append(float(avg_reward))
        history["round_channel_changes"].append(float(avg_channel_changes))
        history["round_num_agents"].append(len(agents))

        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / max(1, communication_rounds // 2))
        print(
            f"Round {round_idx + 1}/{communication_rounds} | "
            f"agents={len(agents)} | reward={avg_reward:.3f} | "
            f"avg_channel_changes={avg_channel_changes:.3f}"
        )

    return history, global_weights


if __name__ == "__main__":
    run_federated_training()
