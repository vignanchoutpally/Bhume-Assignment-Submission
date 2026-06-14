"""Scoring candidate alignments using blurred boundary rasters and imagery edge gradients."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, sobel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io import Village, open_imagery

logger = logging.getLogger(__name__)

class VillageScorer:
    """Precomputes and manages raster layers for scoring candidate alignments."""

    def __init__(self, village: Village, sigma_boundary: float = 2.0, sigma_edge: float = 1.0):
        self.village = village
        
        # 1. Precompute Blurred Boundaries
        if village.boundaries_path:
            logger.info("Precomputing blurred boundary raster...")
            with open_imagery(village.boundaries_path) as src:
                self.bnd_transform = src.transform
                self.bnd_inv_transform = ~src.transform
                bnd_data = src.read(1).astype(np.float32)
                # Apply Gaussian filter to create a continuous potential field
                bnd_blurred = gaussian_filter(bnd_data, sigma=sigma_boundary)
                # Normalize to [0, 1]
                self.bnd_layer = bnd_blurred / 255.0
                self.bnd_shape = self.bnd_layer.shape
        else:
            logger.warning("No boundaries raster found. Boundary score will be zero.")
            self.bnd_layer = None

        # 2. Precompute Imagery Gradients (Edges)
        logger.info("Precomputing imagery edge gradients...")
        with open_imagery(village.imagery_path) as src:
            self.img_transform = src.transform
            self.img_inv_transform = ~src.transform
            
            # Read RGB bands
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
            
            # Convert to grayscale
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            del r, g, b
            
            # Sobel gradients
            dx = sobel(gray, axis=1)
            dy = sobel(gray, axis=0)
            del gray
            
            magnitude = np.hypot(dx, dy)
            del dx, dy
            
            # Smooth the edges slightly
            magnitude_smoothed = gaussian_filter(magnitude, sigma=sigma_edge)
            del magnitude
            
            # Normalize to [0, 1] using 99th percentile to suppress outliers
            p99 = np.percentile(magnitude_smoothed, 99)
            if p99 > 0:
                self.img_layer = np.clip(magnitude_smoothed / p99, 0, 1)
            else:
                self.img_layer = magnitude_smoothed
            self.img_shape = self.img_layer.shape
            
        logger.info("Raster precomputation complete.")

    def score_coordinates(self, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        """Compute the average boundary and imagery edge scores for a set of coordinates in EPSG:3857.

        `xs` and `ys` are 1D arrays of points representing the boundary of the candidate geometry.
        Returns: (boundary_score, edge_score) in [0, 1].
        """
        # 1. Boundary score
        bnd_score = 0.0
        if self.bnd_layer is not None:
            # Convert to pixel coords
            cols, rows = self.bnd_inv_transform * (xs, ys)
            cols = np.round(cols).astype(np.int32)
            rows = np.round(rows).astype(np.int32)
            
            # Filter in-bounds
            mask = (cols >= 0) & (cols < self.bnd_shape[1]) & (rows >= 0) & (rows < self.bnd_shape[0])
            if mask.any():
                bnd_score = float(np.mean(self.bnd_layer[rows[mask], cols[mask]]))

        # 2. Imagery edge score
        cols, rows = self.img_inv_transform * (xs, ys)
        cols = np.round(cols).astype(np.int32)
        rows = np.round(rows).astype(np.int32)
        
        # Filter in-bounds
        mask = (cols >= 0) & (cols < self.img_shape[1]) & (rows >= 0) & (rows < self.img_shape[0])
        edge_score = 0.0
        if mask.any():
            edge_score = float(np.mean(self.img_layer[rows[mask], cols[mask]]))

        return bnd_score, edge_score
