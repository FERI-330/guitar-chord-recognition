"""
Model factory for instantiating different chord recognition models.
"""

import logging
from .base_model import ChordModel
from .chord_model_random_forest import ChordModelRandomForest
from .chord_model_gradient_boost import ChordModelGradientBoosting

logger = logging.getLogger('guitar_chord_recognition')


class ModelFactory:
    """Factory for creating chord recognition models."""

    _MODELS = {
        'random_forest': ChordModelRandomForest,
        'gradient_boost': ChordModelGradientBoosting,
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> ChordModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boost')
            **kwargs: Additional arguments for model initialization

        Returns:
            Model instance
        """
        model_type = model_type.lower()

        if model_type not in cls._MODELS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(cls._MODELS.keys())}")

        model_class = cls._MODELS[model_type]
        model = model_class(**kwargs)

        logger.info(f"Created model: {model_type}")

        return model

    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register a new model class.

        Args:
            model_type: Identifier for the model
            model_class: Model class (must inherit from ChordModel)
        """
        if not issubclass(model_class, ChordModel):
            raise TypeError("Model class must inherit from ChordModel")

        cls._MODELS[model_type] = model_class
        logger.info(f"Registered new model: {model_type}")

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available models."""
        return list(cls._MODELS.keys())
