"""
Example script demonstrating the complete training pipeline.
This is a standalone example that doesn't require the CLI.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.logging_config import setup_logging
from utils.data_loader import DataLoader
from features.feature_extractor import FeatureExtractor
from features.fretboard_detector import FretboardDetector
from evaluation.model_evaluator import ModelEvaluator
from models.model_factory import ModelFactory
from training_pipeline import TrainingPipeline


def example_basic_pipeline():
    """Basic example of using the pipeline."""

    # Setup
    setup_logging(level='INFO')
    logger = __import__('logging').getLogger('guitar_chord_recognition')

    logger.info("Starting basic pipeline example...")

    # Configuration
    config = Config()

    # Initialize pipeline
    pipeline = TrainingPipeline(config, verbose=True)

    try:
        # Load and extract features
        logger.info("Loading training data...")
        X_train, y_train, paths = pipeline.load_and_extract_features(split='train')

        if len(X_train) == 0:
            logger.error("No training data found. Please populate data/ directory.")
            logger.info("Expected structure: data/train/[A-G]/*.jpg")
            return

        logger.info(f"Loaded {len(X_train)} training samples")

        # Split data
        X_train, X_test, y_train, y_test = pipeline.data_loader.create_train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=True
        )

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Train
        logger.info("Training model...")
        pipeline.train_model(X_train, y_train)

        # Evaluate
        logger.info("Evaluating model...")
        results = pipeline.evaluate_model(X_test, y_test)

        # Print results
        logger.info("\n" + "="*60)
        logger.info("RESULTS")
        logger.info("="*60)

        metrics = results['metrics']
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 (Macro): {metrics['f1_macro']:.4f}")

        logger.info("\nPer-Class Performance:")
        for chord, chord_metrics in results['per_class_metrics'].items():
            logger.info(f"  {chord}: F1={chord_metrics['f1']:.4f}")

        logger.info("="*60)
        logger.info("✅ Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


def example_model_selection():
    """Example showing how to select different models."""

    logger = __import__('logging').getLogger('guitar_chord_recognition')

    logger.info("\n" + "="*60)
    logger.info("Model Selection Example")
    logger.info("="*60)

    # Available models
    available = ModelFactory.get_available_models()
    logger.info(f"Available models: {available}")

    # Create different models
    for model_type in available:
        try:
            model = ModelFactory.create_model(model_type)
            logger.info(f"✅ Created {model_type}: {model.get_model_info()}")
        except Exception as e:
            logger.error(f"❌ Could not create {model_type}: {e}")


def example_data_loading():
    """Example showing data loading and preprocessing."""

    logger = __import__('logging').getLogger('guitar_chord_recognition')

    logger.info("\n" + "="*60)
    logger.info("Data Loading Example")
    logger.info("="*60)

    config = Config()
    chord_classes = config.get('data.chord_classes')

    loader = DataLoader(
        chord_classes=chord_classes,
        data_dir='data'
    )

    # Load training data paths
    df_train = loader.load_image_paths(split='train')

    if not df_train.empty:
        logger.info(f"Loaded {len(df_train)} training images")
        logger.info(f"Class distribution:")
        for chord, count in loader.get_class_distribution(df_train['label'].values).items():
            logger.info(f"  {chord}: {count}")
    else:
        logger.info("No training data found")


def example_feature_extraction():
    """Example showing feature extraction."""

    logger = __import__('logging').getLogger('guitar_chord_recognition')

    logger.info("\n" + "="*60)
    logger.info("Feature Extraction Example")
    logger.info("="*60)

    extractor = FeatureExtractor(num_frets=12, num_strings=6)
    detector = FretboardDetector(num_frets=12, num_strings=6)

    logger.info(f"Feature extractor initialized")
    logger.info(f"Fretboard detector initialized")

    # Example feature names
    logger.info(f"Feature names: {extractor.get_feature_names()}")


if __name__ == '__main__':
    """Run all examples."""

    print("\n" + "="*60)
    print("GUITAR CHORD RECOGNITION - EXAMPLE SCRIPTS")
    print("="*60)

    # Setup logging
    setup_logging(level='INFO')
    logger = __import__('logging').getLogger('guitar_chord_recognition')

    # Run examples
    print("\n1. Basic Pipeline Example")
    print("-" * 60)
    example_basic_pipeline()

    print("\n2. Model Selection Example")
    print("-" * 60)
    example_model_selection()

    print("\n3. Data Loading Example")
    print("-" * 60)
    example_data_loading()

    print("\n4. Feature Extraction Example")
    print("-" * 60)
    example_feature_extraction()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
