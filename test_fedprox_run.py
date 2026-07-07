"""Sanity check for FedProx local training path."""

import math
import shutil
from pathlib import Path

from main_v3 import run_federated_training


def main():
    save_dir = Path("experiments/results/_tests/fedprox_sanity")
    if save_dir.is_dir():
        shutil.rmtree(save_dir)

    history, global_weights = run_federated_training(
        communication_rounds=2,
        local_train_steps=2,
        batch_size=4,
        history_length=1,
        number_of_channels=10,
        min_nets=2,
        max_nets=3,
        epsilon_start=0.2,
        epsilon_end=0.05,
        learning_rate=0.00025,
        save_dir=str(save_dir),
        fedprox_mu=0.1,
        run_name="fedprox_sanity",
        seed=321,
    )

    assert len(history["round_rewards"]) == 2, "Expected exactly 2 FedProx rounds."
    assert history["config"]["fedprox_mu"] == 0.1, "FedProx mu was not recorded."
    assert all(math.isfinite(v) for v in history["round_rewards"]), "Rewards must be finite."
    assert isinstance(global_weights, dict) and len(global_weights) > 0, "FedProx did not return weights."
    print("FedProx sanity check passed.")


if __name__ == "__main__":
    main()
