"""Verify FedProx with mu=0 follows the FedAvg code path."""

import shutil
from pathlib import Path

import numpy as np

from main_v3 import run_federated_training


def _run(save_dir, fedprox_mu):
    if save_dir.is_dir():
        shutil.rmtree(save_dir)
    history, _ = run_federated_training(
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
        fedprox_mu=fedprox_mu,
        run_name=f"mu_zero_check_{fedprox_mu}",
        seed=777,
    )
    return history


def main():
    base = Path("experiments/results/_tests")
    fedavg = _run(base / "mu_zero_fedavg", fedprox_mu=0.0)
    fedprox_zero = _run(base / "mu_zero_fedprox", fedprox_mu=0.0)

    np.testing.assert_allclose(fedavg["round_rewards"], fedprox_zero["round_rewards"], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        fedavg["round_channel_changes"],
        fedprox_zero["round_channel_changes"],
        rtol=0.0,
        atol=1e-9,
    )
    print("FedProx mu=0 equivalence check passed.")


if __name__ == "__main__":
    main()
