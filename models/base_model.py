"""
Base model class for guitar chord recognition.
All model implementations should inherit from this class.
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

logger = logging.getLogger('guitar_chord_recognition')


class ChordModel(ABC):
    """Abstract base class for chord recognition models."""

    def __init__(self, model_name: str):
        """
        Initialize model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Test features
            y: Test labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def get_model_info(self) -> str:
        """Get model information."""
        return f"Model: {self.model_name}, Trained: {self.is_trained}"
