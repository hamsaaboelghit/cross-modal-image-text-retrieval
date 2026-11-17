
import importlib
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODULES = [
    # Packages
    "src",
    "src.data",
    "src.models",
    "src.training",
    "src.evaluation",
    "src.retrieval",

    # Data
    "src.data.flickr30k_dataset",

    # Models
    "src.models.attention",
    "src.models.image_encoder",
    "src.models.text_encoder",
    "src.models.projection_head",
    "src.models.cross_modal_attention",
    "src.models.full_model",
    "src.models.baseline_model",

    # Training
    "src.training.contrastive_loss",
    "src.training.triplet_loss",
    "src.training.margin_loss",
    "src.training.train_config",
    "src.training.optimizer",
    "src.training.trainer",
    "src.training.evaluator",
    "src.training.checkpoint_manager",

    # Evaluation
    "src.evaluation.retrieval_metrics",
    "src.evaluation.retrieval_evaluator",
    "src.evaluation.evaluation_runner",

    # Retrieval
    "src.retrieval.retrieval_engine",
]


class TestPackageImports(unittest.TestCase):

    def test_all_modules_import(self):
        failed_modules = []

        for module_name in MODULES:
            try:
                importlib.import_module(module_name)
            except Exception as error:
                failed_modules.append(
                    (module_name, repr(error))
                )

        if failed_modules:
            details = "\n".join(
                f"{name}: {error}"
                for name, error in failed_modules
            )

            self.fail(
                "Some project modules could not be imported:\n"
                + details
            )

    def test_tensorflow_is_available(self):
        import tensorflow as tf

        self.assertTrue(
            tf.__version__.startswith("2.")
        )

    def test_core_model_classes(self):
        from src.models.full_model import CrossModalRetrievalModel
        from src.models.baseline_model import BaselineRetrievalModel

        self.assertTrue(
            callable(CrossModalRetrievalModel)
        )

        self.assertTrue(
            callable(BaselineRetrievalModel)
        )

    def test_core_loss_classes(self):
        from src.training.contrastive_loss import MultiPositiveInfoNCELoss
        from src.training.triplet_loss import TripletContrastiveLoss
        from src.training.margin_loss import MarginBasedContrastiveLoss

        self.assertTrue(callable(MultiPositiveInfoNCELoss))
        self.assertTrue(callable(TripletContrastiveLoss))
        self.assertTrue(callable(MarginBasedContrastiveLoss))


if __name__ == "__main__":
    unittest.main(verbosity=2)
