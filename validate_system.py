"""
Validation script to verify all modules load correctly.
Run this after installation to ensure everything is working.
"""

import sys
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
from utils.logging_config import setup_logging
logger = setup_logging()

def validate_imports():
    """Validate that all modules can be imported."""

    tests = []

    try:
        logger.info("Testing imports...")

        # Utils
        from utils.config import Config
        tests.append(("Config", "✅"))

        from utils.logging_config import setup_logging
        tests.append(("Logging", "✅"))

        from utils.data_loader import DataLoader
        tests.append(("DataLoader", "✅"))

        # Features
        from features.feature_extractor import FeatureExtractor
        tests.append(("FeatureExtractor", "✅"))

        from features.fretboard_detector import FretboardDetector
        tests.append(("FretboardDetector", "✅"))

        # Models
        from models.base_model import ChordModel
        tests.append(("BaseModel", "✅"))

        from models.chord_model_random_forest import ChordModelRandomForest
        tests.append(("RandomForestModel", "✅"))

        from models.chord_model_gradient_boost import ChordModelGradientBoosting
        tests.append(("GradientBoostModel", "✅"))

        from models.model_factory import ModelFactory
        tests.append(("ModelFactory", "✅"))

        # Evaluation
        from evaluation.model_evaluator import ModelEvaluator
        tests.append(("ModelEvaluator", "✅"))

        # Pipeline
        from training_pipeline import TrainingPipeline
        tests.append(("TrainingPipeline", "✅"))

    except Exception as e:
        tests.append(("ERROR", f"❌ {e}"))
        logger.error(f"Import validation failed: {e}")
        return False

    # Print results
    logger.info("\n" + "=" * 50)
    logger.info("IMPORT VALIDATION RESULTS")
    logger.info("=" * 50)

    for module, status in tests:
        logger.info(f"{module:<30} {status}")

    logger.info("=" * 50)

    if all(status == "✅" for _, status in tests):
        logger.info("✅ All imports validated successfully!")
        return True
    else:
        logger.error("❌ Some imports failed")
        return False


def validate_models():
    """Validate that models can be instantiated."""

    logger.info("\nValidating model instantiation...")

    from models.model_factory import ModelFactory

    try:
        # Test Random Forest
        rf_model = ModelFactory.create_model(
            'random_forest',
            n_estimators=10
        )
        logger.info("✅ Random Forest model instantiated")

        # Test Gradient Boosting
        gb_model = ModelFactory.create_model(
            'gradient_boost',
            n_estimators=10
        )
        logger.info("✅ Gradient Boosting model instantiated")

        return True

    except Exception as e:
        logger.error(f"❌ Model instantiation failed: {e}")
        return False


def validate_config():
    """Validate configuration system."""

    logger.info("\nValidating configuration system...")

    from utils.config import Config

    try:
        config = Config()

        # Test getters
        model_type = config.get('model.type')
        logger.info(f"✅ Config getter works: model.type = {model_type}")

        # Test setters
        config.set('model.type', 'test_value')
        assert config.get('model.type') == 'test_value'
        logger.info("✅ Config setter works")

        # Test default values
        assert config.get('nonexistent.key', 'default') == 'default'
        logger.info("✅ Config defaults work")

        return True

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        return False


if __name__ == '__main__':
    logger.info("Starting system validation...")

    results = []
    results.append(("Imports", validate_imports()))
    results.append(("Config", validate_config()))
    results.append(("Models", validate_models()))

    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:<20} {status}")

    logger.info("=" * 50)

    if all(passed for _, passed in results):
        logger.info("✅ All validations passed! System is ready to use.")
        sys.exit(0)
    else:
        logger.error("❌ Some validations failed. Please check the errors above.")
        sys.exit(1)
