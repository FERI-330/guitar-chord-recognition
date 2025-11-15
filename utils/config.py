"""
Configuration module for guitar chord recognition system.
Handles loading and managing configuration parameters.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for guitar chord recognition system."""

    # Default configuration
    DEFAULTS = {
        "data": {
            "train_dir": "data/train",
            "test_dir": "data/test",
            "chord_classes": ["A", "B", "C", "D", "E", "F", "G"],
            "image_extensions": [".jpg", ".png", ".jpeg"],
        },
        "feature_extraction": {
            "hand_detection_method": "mediapipe",
            "fretboard_detection_enabled": True,
            "feature_normalization": True,
            "fret_positions": 12,  # Approximate 12-fret detection
            "string_count": 6,
        },
        "model": {
            "type": "random_forest",  # Options: "random_forest", "cnn", "gradient_boost"
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "cross_validation_folds": 5,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "random_state": 42,
        },
        "cnn": {
            "input_size": 224,
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "early_stopping_patience": 5,
        },
        "evaluation": {
            "metrics": ["accuracy", "f1_macro", "roc_auc_ovr", "precision", "recall"],
            "report_per_class": True,
            "save_plots": True,
            "plot_dir": "plots",
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs",
            "log_file": "guitar_chord_recognition.log",
        },
    }

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to JSON configuration file. If None, uses defaults.
        """
        self.config = self.DEFAULTS.copy()

        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
            logger.info(f"Configuration loaded from {config_file}")
        else:
            logger.info("Using default configuration")

    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)

            # Deep merge custom config with defaults
            self._merge_dicts(self.config, custom_config)
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            raise

    @staticmethod
    def _merge_dicts(target: Dict, source: Dict) -> None:
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                Config._merge_dicts(target[key], value)
            else:
                target[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Path to config value (e.g., 'model.type')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self.config.copy()

    def save(self, output_file: str) -> None:
        """Save configuration to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving config to {output_file}: {e}")
            raise
