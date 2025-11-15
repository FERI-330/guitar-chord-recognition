"""
Training pipeline for guitar chord recognition.
Orchestrates data loading, feature extraction, model training, and evaluation.
"""

import logging
import numpy as np
import pandas as pd
import cv2
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

from utils.data_loader import DataLoader
from utils.logging_config import setup_logging
from features.feature_extractor import FeatureExtractor
from features.fretboard_detector import FretboardDetector
from evaluation.model_evaluator import ModelEvaluator
from models.model_factory import ModelFactory

logger = logging.getLogger('guitar_chord_recognition')


class TrainingPipeline:
    """Orchestrates the complete training pipeline."""

    def __init__(self, config, verbose: bool = True):
        """
        Initialize training pipeline.

        Args:
            config: Configuration object
            verbose: Whether to print progress
        """
        self.config = config
        self.verbose = verbose

        # Initialize components
        self.data_loader = None
        self.feature_extractor = None
        self.fretboard_detector = None
        self.evaluator = None
        self.model = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        chord_classes = self.config.get('data.chord_classes')

        self.data_loader = DataLoader(
            chord_classes=chord_classes,
            data_dir=self.config.get('data.train_dir', 'data/train').rsplit('/', 1)[0]
        )

        num_frets = self.config.get('feature_extraction.fret_positions', 12)
        num_strings = self.config.get('feature_extraction.string_count', 6)

        self.feature_extractor = FeatureExtractor(
            num_frets=num_frets,
            num_strings=num_strings
        )

        self.fretboard_detector = FretboardDetector(
            num_frets=num_frets,
            num_strings=num_strings
        )

        self.evaluator = ModelEvaluator(
            classes=chord_classes,
            save_plots=self.config.get('evaluation.save_plots', True),
            plot_dir=self.config.get('evaluation.plot_dir', 'plots')
        )

        logger.info("Pipeline components initialized")

    def load_and_extract_features(self, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load images and extract features.

        Args:
            split: 'train' or 'test'

        Returns:
            Tuple of (features, labels, image_paths)
        """
        # Load image paths
        df_images = self.data_loader.load_image_paths(split=split)

        if df_images.empty:
            logger.error(f"No images found for {split} split")
            return np.array([]), np.array([]), []

        features_list = []
        labels_list = []
        image_paths = []

        logger.info(f"Extracting features from {len(df_images)} images")

        for idx, row in df_images.iterrows():
            try:
                # Load image
                image = cv2.imread(row['path'])
                if image is None:
                    logger.warning(f"Could not read image: {row['path']}")
                    continue

                # Get fretboard grid
                fret_positions, string_positions, _ = self.fretboard_detector.get_fretboard_grid(image)

                # Extract features
                features = self.feature_extractor.extract_features_from_image(
                    image,
                    fret_positions=fret_positions,
                    string_positions=string_positions
                )

                if features is None:
                    logger.warning(f"Could not extract features from: {row['path']}")
                    continue

                features_list.append(features)
                labels_list.append(row['label'])
                image_paths.append(row['path'])

                if self.verbose and (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1} images")

            except Exception as e:
                logger.error(f"Error processing {row['path']}: {e}")
                continue

        if not features_list:
            logger.error("No features were successfully extracted")
            return np.array([]), np.array([]), []

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"Extracted features: X.shape={X.shape}, y.shape={y.shape}")

        return X, y, image_paths

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Normalize features
        X_train = self.data_loader.normalize_features(X_train, fit=True)

        # Encode labels
        y_train_encoded = self.data_loader.encode_labels(y_train, fit=True)

        # Create model
        model_type = self.config.get('model.type', 'random_forest')
        model_config = self.config.get(model_type, {})

        self.model = ModelFactory.create_model(model_type, **model_config)

        # Train
        logger.info(f"Training model: {model_type}")
        self.model.fit(X_train, y_train_encoded)

        logger.info("Model training completed")

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Normalize using fitted scaler
        X_test = self.data_loader.normalize_features(X_test, fit=False)

        # Encode labels
        y_test_encoded = self.data_loader.encode_labels(y_test, fit=False)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Multiclass metrics
        metrics = self.evaluator.evaluate_multiclass(y_test_encoded, y_pred, y_pred_proba)

        # Per-class metrics
        per_class_metrics = self.evaluator.evaluate_per_class(y_test_encoded, y_pred, y_pred_proba)

        # Classification report
        report = self.evaluator.get_classification_report(y_test_encoded, y_pred)
        logger.info(f"\nClassification Report:\n{report}")

        # Save plots
        self.evaluator.plot_confusion_matrix(y_test_encoded, y_pred, "confusion_matrix.png")
        self.evaluator.plot_roc_curves(y_test_encoded, y_pred_proba, "roc_curves.png")
        self.evaluator.plot_metrics_summary(per_class_metrics, "metrics_summary.png")

        return {
            'metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'classification_report': report
        }
