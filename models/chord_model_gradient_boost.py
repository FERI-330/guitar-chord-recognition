"""
Gradient Boosting model for guitar chord recognition.
"""

import logging
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from .base_model import ChordModel

logger = logging.getLogger('guitar_chord_recognition')


class ChordModelGradientBoosting(ChordModel):
    """Gradient Boosting implementation for chord recognition."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_samples_split: int = 5,
        random_state: int = 42
    ):
        """
        Initialize Gradient Boosting model.

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate (shrinkage)
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split a node
            random_state: Random seed
        """
        super().__init__(model_name="GradientBoosting")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        # Initialize model
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            verbose=1
        )

        logger.info(f"Initialized {self.model_name} with {n_estimators} stages")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Gradient Boosting model.

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
