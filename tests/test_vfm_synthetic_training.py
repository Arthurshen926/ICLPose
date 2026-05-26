from feature_extract.vfm.training import SyntheticSelectorTrainingConfig, run_synthetic_selector_training


def test_synthetic_selector_training_learns_signal_group():
    result = run_synthetic_selector_training(
        SyntheticSelectorTrainingConfig(
            steps=80,
            batch_size=32,
            input_dim=16,
            output_dim=8,
            group_size=4,
            candidates_per_query=4,
            seed=0,
            device="cpu",
        )
    )

    assert result.final_loss < result.initial_loss
    assert result.final_top1_acc > 0.8
    assert result.signal_group_gate > result.noise_group_gate_mean
