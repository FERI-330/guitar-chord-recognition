"""
Random Forest model for guitar chord recognition.
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import ChordModel

logger = logging.getLogger('guitar_chord_recognition')


class ChordModelRandomForest(ChordModel):
    """Random Forest implementation for chord recognition."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 5,
        random_state: int = 42
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split a node
            random_state: Random seed
        """
        super().__init__(model_name="RandomForest")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

        logger.info(f"Initialized {self.model_name} with {n_estimators} trees")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model.

        Args:
            X: Training features
            y: Training labels
        """
        try:
            logger.info(f"Training {self.model_name} on {len(X)} samples")
            self.model.fit(X, y)
            self.is_trained = True

            # Feature importance
            feature_importance = self.model.feature_importances_
            logger.info(f"Mean feature importance: {feature_importance.mean():.4f}")

        except Exception as e:
            logger.error(f"Error training {self.model_name}: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} model not trained yet")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Probability matrix
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} model not trained yet")

        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names: list = None):
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        importance = self.model.feature_importances_

        if feature_names:
            return dict(zip(feature_names, importance))
        else:
            return importance
