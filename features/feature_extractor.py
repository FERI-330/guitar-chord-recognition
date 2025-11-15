"""
Feature extraction module for guitar chord recognition.
Extracts hand landmarks and fretboard features.
"""

import logging
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger('guitar_chord_recognition')


class FeatureExtractor:
    """Extracts features from images for chord recognition."""

    def __init__(self, num_frets: int = 12, num_strings: int = 6):
        """
        Initialize FeatureExtractor.

        Args:
            num_frets: Approximate number of frets
            num_strings: Number of guitar strings
        """
        self.num_frets = num_frets
        self.num_strings = num_strings

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.feature_names = []

    def extract_hand_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """
        Extract hand landmarks using MediaPipe.

        Args:
            image: Input image (BGR)

        Returns:
            List of (x, y, z) coordinates for each landmark or None
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Invalid image for hand landmark extraction")
                return None

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image
            results = self.hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                logger.debug("No hand landmarks detected")
                return None

            # Extract first hand
            hand = results.multi_hand_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]

            logger.debug(f"Extracted {len(landmarks)} hand landmarks")

            return landmarks

        except Exception as e:
            logger.error(f"Error in hand landmark extraction: {e}")
            return None

    def compute_fingertip_distances(
        self,
        landmarks: List[Tuple[float, float, float]],
        wrist_idx: int = 0
    ) -> np.ndarray:
        """
        Compute distances from wrist to fingertips.

        Args:
            landmarks: Hand landmarks
            wrist_idx: Index of wrist landmark (default 0)

        Returns:
            Array of distances [thumb, index, middle, ring, pinky]
        """
        if not landmarks or len(landmarks) < 21:
            logger.warning("Insufficient landmarks for distance computation")
            return np.zeros(5)

        # Fingertip indices in MediaPipe: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        fingertip_indices = [4, 8, 12, 16, 20]

        wrist = np.array(landmarks[wrist_idx][:2])
        distances = []

        for idx in fingertip_indices:
            fingertip = np.array(landmarks[idx][:2])
            dist = np.linalg.norm(fingertip - wrist)
            distances.append(dist)

        return np.array(distances)

    def compute_finger_angles(
        self,
        landmarks: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Compute angles for each finger.

        Args:
            landmarks: Hand landmarks

        Returns:
            Array of finger angles
        """
        if not landmarks or len(landmarks) < 21:
            logger.warning("Insufficient landmarks for angle computation")
            return np.zeros(5)

        # Finger key points: (base, middle, tip)
        fingers = [
            (2, 3, 4),    # Thumb
            (5, 6, 7, 8),   # Index
            (9, 10, 11, 12),  # Middle
            (13, 14, 15, 16), # Ring
            (17, 18, 19, 20)  # Pinky
        ]

        angles = []

        for finger_points in fingers:
            # Use last 3 points for angle calculation
            p1 = np.array(landmarks[finger_points[-3]][:2])
            p2 = np.array(landmarks[finger_points[-2]][:2])
            p3 = np.array(landmarks[finger_points[-1]][:2])

            # Compute angle at p2
            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            angles.append(angle)

        return np.array(angles)

    def compute_hand_position(
        self,
        landmarks: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Compute hand position features (centroid, spread).

        Args:
            landmarks: Hand landmarks

        Returns:
            Array of position features [cx, cy, spread]
        """
        if not landmarks:
            logger.warning("No landmarks for hand position computation")
            return np.zeros(3)

        coords = np.array([[lm[0], lm[1]] for lm in landmarks])

        # Centroid
        centroid = coords.mean(axis=0)

        # Spread (std of coordinates)
        spread = np.std(coords)

        return np.array([centroid[0], centroid[1], spread])

    def extract_features_from_image(
        self,
        image: np.ndarray,
        fret_positions: Optional[np.ndarray] = None,
        string_positions: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Extract all features from a single image.

        Args:
            image: Input image (BGR)
            fret_positions: Approximate fret positions (optional)
            string_positions: Approximate string positions (optional)

        Returns:
            Feature vector or None if extraction fails
        """
        try:
            # Extract hand landmarks
            landmarks = self.extract_hand_landmarks(image)

            if landmarks is None:
                logger.warning("Could not extract hand landmarks")
                return None

            # Compute hand features
            fingertip_distances = self.compute_fingertip_distances(landmarks)
            finger_angles = self.compute_finger_angles(landmarks)
            hand_position = self.compute_hand_position(landmarks)

            # Combine all features
            features = np.concatenate([
                fingertip_distances,
                finger_angles,
                hand_position
            ])

            # Optional: Add fretboard features if provided
            if fret_positions is not None and string_positions is not None:
                # Relative hand position to fretboard
                h, w = image.shape[:2]
                fret_feature = np.min(np.abs(landmarks[8][0] * w - fret_positions))  # Index finger to nearest fret
                string_feature = np.min(np.abs(landmarks[8][1] * h - string_positions))  # Index finger to nearest string

                features = np.concatenate([features, [fret_feature, string_feature]])

            logger.debug(f"Extracted feature vector of size {len(features)}")

            if not self.feature_names:
                self._init_feature_names(len(features) - 2 if fret_positions is not None else len(features))

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def _init_feature_names(self, total_size: int) -> None:
        """Initialize feature names."""
        names = []
        names.extend([f"fingertip_dist_{i}" for i in range(5)])
        names.extend([f"finger_angle_{i}" for i in range(5)])
        names.extend(["hand_pos_x", "hand_pos_y", "hand_spread"])

        if len(names) < total_size:
            names.extend(["fret_proximity", "string_proximity"])

        self.feature_names = names[:total_size]

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return self.feature_names.copy()
