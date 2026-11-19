"""Tests for the Baseline-only OFAT hyperparameter experiment design."""

import unittest

from src.training.run_hyperparameter_experiments import (
    REFERENCE,
    build_experiments,
    build_run_metadata,
    build_result_row,
    validate_experiment_design,
    validate_resume_metadata,
)


class HyperparameterExperimentDesignTests(unittest.TestCase):
    def test_builds_reference_plus_nine_variations(self):
        experiments = build_experiments()
        self.assertEqual(len(experiments), 10)
        self.assertEqual(experiments[0]["name"], "reference")
        self.assertEqual(
            {key: experiments[0][key] for key in REFERENCE},
            REFERENCE,
        )

    def test_each_variation_changes_exactly_one_parameter(self):
        experiments = build_experiments()
        validate_experiment_design(experiments)

        for experiment in experiments[1:]:
            changed = [
                key
                for key, reference_value in REFERENCE.items()
                if experiment[key] != reference_value
            ]
            self.assertEqual(changed, [experiment["changed_parameter"]])

    def test_expected_search_values_are_present(self):
        experiments = build_experiments()
        observed = {
            parameter: {
                item[parameter]
                for item in experiments
            }
            for parameter in REFERENCE
        }

        self.assertEqual(observed["learning_rate"], {1e-5, 1e-4, 1e-3})
        self.assertEqual(observed["batch_size"], {8, 16, 32})
        self.assertEqual(observed["output_dim"], {256, 512})
        self.assertEqual(observed["dropout"], {0.1, 0.2, 0.3})
        self.assertEqual(observed["temperature"], {0.05, 0.07, 0.1})

    def test_run_metadata_records_baseline_and_validation_scope(self):
        metadata = build_run_metadata(build_experiments(), epochs=5)
        self.assertEqual(metadata["model_type"], "baseline")
        self.assertEqual(metadata["selection_split"], "validation")
        self.assertEqual(metadata["epochs"], 5)
        self.assertEqual(len(metadata["experiments"]), 10)

    def test_result_row_records_requested_epoch_count(self):
        experiment = build_experiments()[0]
        history = {
            "train_loss": [1.2, 0.9, 0.7],
            "train_retrieval_loss": [1.2, 0.9, 0.7],
            "val_retrieval_loss": [1.1, 0.8, 0.85],
        }
        metrics = {
            "overall_mean_recall": 0.42,
            "overall_mrr": 0.31,
            "evaluation_split": "validation",
            "test_set_used": False,
            "best_val_retrieval_loss": 0.8,
            "checkpoint_path": "checkpoints/mock",
        }

        row = build_result_row(
            experiment=experiment,
            history=history,
            metrics=metrics,
            duration=1.5,
            epochs_requested=12,
        )

        self.assertEqual(row["epochs_requested"], 12)
        self.assertEqual(row["epochs_completed"], 3)
        self.assertEqual(row["best_epoch"], 2)

    def test_resume_rejects_different_run_settings(self):
        expected = build_run_metadata(build_experiments(), epochs=5)
        incompatible = dict(expected)
        incompatible["epochs"] = 10

        with self.assertRaises(ValueError):
            validate_resume_metadata(incompatible, expected)


if __name__ == "__main__":
    unittest.main()
