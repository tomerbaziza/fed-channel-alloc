"""Verify persistent local replay grows across FL communication rounds."""

from main_v3 import run_federated_training


def main():
    history, _ = run_federated_training(
        communication_rounds=15,
        local_train_steps=1,
        fedprox_mu=0.01,
        batch_size=32,
        replay_memory_size=10000,
        persistent_replay=True,
        save_dir="experiments/results/_tests/persistent_replay_check",
        run_name="persistent_replay_check",
        seed=42,
        checkpoint_every=0,
    )

    counts = history["replay_buffer_counts"]
    print("\n=== Replay buffer growth ===")
    for i, count in enumerate(counts, start=1):
        print(f"  round {i:2d}: replay_count={count}")

    assert counts[-1] > counts[0], "Replay buffer should grow across rounds."
    assert counts[-1] >= 32, f"Expected replay_count >= batch_size (32), got {counts[-1]}"
    print(f"\nPASS: final replay_count={counts[-1]} (batch_size=32)")


if __name__ == "__main__":
    main()
