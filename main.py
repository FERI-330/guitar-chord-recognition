"""
Main entry point for guitar chord recognition system.
Supports both CLI and batch processing modes.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from utils.logging_config import setup_logging
from training_pipeline import TrainingPipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Guitar Chord Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python main.py --train

  # Train with custom config
  python main.py --train --config config.json

  # Train specific model
  python main.py --train --model random_forest

  # Cross-validation
  python main.py --train --cv 5
        """
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate trained model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (JSON)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'gradient_boost'],
        default='random_forest',
        help='Model type to use'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=None,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(level=log_level)
    logger = logging.getLogger('guitar_chord_recognition')

    logger.info("=" * 60)
    logger.info("Guitar Chord Recognition System")
    logger.info("=" * 60)

    # Load configuration
    config = Config(config_file=args.config)
    if args.model:
        config.set('model.type', args.model)
    if args.cv:
        config.set('model.cross_validation_folds', args.cv)

    logger.info(f"Configuration loaded: model={config.get('model.type')}")

    # Initialize pipeline
    try:
        pipeline = TrainingPipeline(config, verbose=args.verbose)

        if args.train:
            logger.info("Starting training pipeline...")

            # Load and extract features
            X_train, y_train, _ = pipeline.load_and_extract_features(split='train')

            if len(X_train) == 0:
                logger.error("No training data available")
                sys.exit(1)

            # Create train-test split
            test_size = config.get('model.test_size', 0.2)
            stratify = config.get('model.stratify', True)

            X_train, X_test, y_train, y_test = pipeline.data_loader.create_train_test_split(
                X_train, y_train,
                test_size=test_size,
                stratify=stratify
            )

            # Train model
            pipeline.train_model(X_train, y_train)

            # Evaluate
            logger.info("Evaluating model...")
            results = pipeline.evaluate_model(X_test, y_test)

            # Print results
            logger.info("\n" + "=" * 60)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 60)

            metrics = results['metrics']
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 (Macro): {metrics['f1_macro']:.4f}")
            logger.info(f"F1 (Weighted): {metrics['f1_weighted']:.4f}")

            if 'roc_auc_ovr' in metrics:
                logger.info(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")

            logger.info("\nPer-Class Metrics:")
            for class_name, class_metrics in results['per_class_metrics'].items():
                logger.info(f"  {class_name}: "
                          f"Precision={class_metrics['precision']:.4f}, "
                          f"Recall={class_metrics['recall']:.4f}, "
                          f"F1={class_metrics['f1']:.4f}")

            logger.info("=" * 60)
            logger.info("✅ Training and evaluation completed successfully!")
            logger.info(f"Plots saved to: plots/")
            logger.info(f"Logs saved to: logs/")

        elif args.evaluate:
            logger.info("Evaluation mode not implemented yet")

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
