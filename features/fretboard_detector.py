"""
Fretboard detection module.
Handles fretboard localization and fret position approximation.
"""

import logging
import cv2
import numpy as np
from typing import Tuple, Optional, List

logger = logging.getLogger('guitar_chord_recognition')


class FretboardDetector:
    """Detects fretboard outline and approximates fret positions."""

    def __init__(self, num_frets: int = 12, num_strings: int = 6):
        """
        Initialize FretboardDetector.

        Args:
            num_frets: Approximate number of frets to detect
            num_strings: Number of guitar strings (typically 6)
        """
        self.num_frets = num_frets
        self.num_strings = num_strings

    def detect_fretboard_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect fretboard region using edge detection and contours.

        Args:
            image: Input image (BGR)

        Returns:
            Bounding box [x, y, w, h] or None if not found
        """
        if image is None or image.size == 0:
            logger.warning("Invalid image for fretboard detection")
            return None

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges to connect nearby lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.debug("No contours found for fretboard")
                return None

            # Find largest rectangular contour (likely the fretboard)
            largest_contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest_contour)

            logger.debug(f"Fretboard region detected: {bbox}")

            return np.array(bbox)

        except Exception as e:
            logger.error(f"Error in fretboard detection: {e}")
            return None

    def detect_fret_lines(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Detect fret lines using Hough transform.

        Args:
            image: Input image (BGR)

        Returns:
            List of fret line positions (x-coordinates) or None
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Detect vertical lines (frets)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                logger.debug("No fret lines detected")
                return None

            # Extract x-coordinates of vertical lines
            fret_positions = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is approximately vertical
                if abs(x2 - x1) < 5:
                    fret_positions.append((x1 + x2) / 2)

            fret_positions = sorted(list(set([int(p) for p in fret_positions])))
            logger.debug(f"Detected {len(fret_positions)} potential fret lines")

            return fret_positions if fret_positions else None

        except Exception as e:
            logger.error(f"Error in fret line detection: {e}")
            return None

    def detect_string_lines(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Detect string lines using Hough transform.

        Args:
            image: Input image (BGR)

        Returns:
            List of string line positions (y-coordinates) or None
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Detect horizontal lines (strings)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                logger.debug("No string lines detected")
                return None

            # Extract y-coordinates of horizontal lines
            string_positions = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is approximately horizontal
                if abs(y2 - y1) < 5:
                    string_positions.append((y1 + y2) / 2)

            string_positions = sorted(list(set([int(p) for p in string_positions])))
            logger.debug(f"Detected {len(string_positions)} potential string lines")

            return string_positions if string_positions else None

        except Exception as e:
            logger.error(f"Error in string line detection: {e}")
            return None

    def approximate_fret_positions(
        self,
        image: np.ndarray,
        fretboard_bbox: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Approximate fret positions if not all are visible.

        Args:
            image: Input image
            fretboard_bbox: Bounding box of fretboard [x, y, w, h]

        Returns:
            Array of fret x-positions
        """
        fret_lines = self.detect_fret_lines(image)

        if fret_lines is None or len(fret_lines) < 2:
            # Fallback: approximate uniform spacing
            if fretboard_bbox is not None:
                x, y, w, h = fretboard_bbox
                fret_positions = np.linspace(x, x + w, self.num_frets)
            else:
                h, w = image.shape[:2]
                fret_positions = np.linspace(0, w, self.num_frets)

            logger.debug("Using approximated fret positions (uniform spacing)")
        else:
            fret_positions = np.array(fret_lines)

        return fret_positions

    def approximate_string_positions(
        self,
        image: np.ndarray,
        fretboard_bbox: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Approximate string positions if not all are visible.

        Args:
            image: Input image
            fretboard_bbox: Bounding box of fretboard [x, y, w, h]

        Returns:
            Array of string y-positions
        """
        string_lines = self.detect_string_lines(image)

        if string_lines is None or len(string_lines) < 2:
            # Fallback: approximate uniform spacing
            if fretboard_bbox is not None:
                x, y, w, h = fretboard_bbox
                string_positions = np.linspace(y, y + h, self.num_strings)
            else:
                h, w = image.shape[:2]
                string_positions = np.linspace(0, h, self.num_strings)

            logger.debug("Using approximated string positions (uniform spacing)")
        else:
            string_positions = np.array(string_lines)

        return string_positions

    def get_fretboard_grid(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get fretboard grid (fret and string positions).

        Args:
            image: Input image

        Returns:
            Tuple of (fret_positions, string_positions, fretboard_bbox)
        """
        fretboard_bbox = self.detect_fretboard_region(image)

        fret_positions = self.approximate_fret_positions(image, fretboard_bbox)
        string_positions = self.approximate_string_positions(image, fretboard_bbox)

        logger.debug(f"Fretboard grid: {len(fret_positions)} frets, {len(string_positions)} strings")

        return fret_positions, string_positions, fretboard_bbox
