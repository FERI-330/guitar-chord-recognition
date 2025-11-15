"""
Data loading and preprocessing module for guitar chord recognition.
Handles image loading, feature extraction, and data splitting.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger('guitar_chord_recognition')


class DataLoader:
    """Handles data loading and preprocessing for guitar chord recognition."""

    def __init__(self, chord_classes: List[str], data_dir: str = "data"):
        """
        Initialize DataLoader.

        Args:
            chord_classes: List of chord class labels
            data_dir: Root data directory
        """
        self.chord_classes = chord_classes
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_image_paths(self, split: str = "train") -> pd.DataFrame:
        """
        Load image paths and labels from directory structure.

        Args:
            split: "train" or "test"

        Returns:
            DataFrame with 'path' and 'label' columns
        """
        image_records = []
        split_dir = Path(self.data_dir) / split

        if not split_dir.exists():
            logger.warning(f"Directory {split_dir} does not exist")
            return pd.DataFrame()

        for chord_class in self.chord_classes:
            class_dir = split_dir / chord_class

            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} not found")
                continue

            # Get all image files
            for ext in ['.jpg', '.png', '.jpeg']:
                for img_path in class_dir.glob(f'*{ext}'):
                    image_records.append({
                        'path': str(img_path),
                        'label': chord_class
                    })

        if not image_records:
            logger.warning(f"No images found in {split_dir}")
            return pd.DataFrame()

        df = pd.DataFrame(image_records)
        logger.info(f"Loaded {len(df)} image paths from {split_dir}")

        return df

    def encode_labels(self, labels: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Encode string labels to integers.

        Args:
            labels: Array of string labels
            fit: Whether to fit the encoder (True for training, False for test)

        Returns:
            Encoded labels
        """
        if fit:
            self.label_encoder.fit(self.chord_classes)
            logger.info(f"Label encoder fitted with classes: {self.chord_classes}")

        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(encoded_labels)

    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler.

        Args:
            X: Feature array
            fit: Whether to fit the scaler (True for training, False for test)

        Returns:
            Normalized features
        """
        if fit:
            X_normalized = self.scaler.fit_transform(X)
            logger.info("Scaler fitted and features normalized")
        else:
            X_normalized = self.scaler.transform(X)
            logger.info("Features normalized using fitted scaler")

        return X_normalized

    def create_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train-test split.

        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to use stratified split

        Returns:
            X_train, X_test, y_train, y_test
        """
        stratify_arg = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )

        logger.info(f"Train/test split: {len(X_train)}/{len(X_test)} "
                   f"(stratified={stratify})")

        return X_train, X_test, y_train, y_test

    def create_kfold_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        random_state: int = 42
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create k-fold cross-validation splits.

        Args:
            X: Features
            y: Labels
            n_splits: Number of folds
            random_state: Random seed

        Returns:
            List of (train_indices, test_indices) tuples
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            splits.append((train_idx, test_idx))
            logger.debug(f"Fold {fold + 1}: train={len(train_idx)}, test={len(test_idx)}")

        logger.info(f"Created {n_splits}-fold stratified cross-validation splits")

        return splits

    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get distribution of classes in labels."""
        unique, counts = np.unique(y, return_counts=True)
        distribution = {self.label_encoder.inverse_transform([label])[0]: int(count)
                       for label, count in zip(unique, counts)}
        return distribution
