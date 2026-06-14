"""Local alignment engine generating and optimizing candidate translations and rotations."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io import Village
from src.scoring import VillageScorer

logger = logging.getLogger(__name__)

def sample_points_on_boundary(geom, step: float = 1.0) -> np.ndarray:
    """Sample points at regular metric intervals along a polygon boundary (EPSG:3857)."""
    points = []
    if geom.geom_type == 'Polygon':
        ex = geom.exterior
        d = 0.0
        while d < ex.length:
            pt = ex.interpolate(d)
            points.append((pt.x, pt.y))
            d += step
        # Append the last coordinate to close it
        points.append(ex.coords[-1])
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            ex = poly.exterior
            d = 0.0
            while d < ex.length:
                pt = ex.interpolate(d)
                points.append((pt.x, pt.y))
                d += step
            points.append(ex.coords[-1])
    else:
        # Fallback to centroid if not polygon
        points.append((geom.centroid.x, geom.centroid.y))
        
    return np.array(points)


def find_plot_neighbors(plots_utm: gpd.GeoDataFrame) -> dict[str, list[str]]:
    """Build a neighbor adjacency map using spatial index intersections (within 0.5m distance)."""
    neighbors = {}
    spatial_index = plots_utm.sindex
    
    for pn in plots_utm.index:
        geom = plots_utm.loc[pn, 'geometry']
        # Intersects bounds
        possible_idx = list(spatial_index.intersection(geom.bounds))
        possible_plots = plots_utm.iloc[possible_idx]
        
        actual = []
        for opn, ogeom in zip(possible_plots.index, possible_plots.geometry):
            if opn != pn and geom.distance(ogeom) < 0.5:
                actual.append(opn)
        neighbors[pn] = actual
        
    return neighbors


class CandidateSearchEngine:
    """Generates candidate transforms and evaluates them using vectorized grid lookups."""

    def __init__(self, scorer: VillageScorer):
        self.scorer = scorer
        self.village = scorer.village
        
        # Build spatial representations in metric projection
        lon = self.village.plots.geometry.iloc[0].centroid.x
        self.utm_crs = f'EPSG:{32600 + int((lon + 180) // 6) + 1}'
        self.plots_utm = self.village.plots.to_crs(self.utm_crs)
        
        # Reproject plots to EPSG:3857 for direct raster lookup
        self.plots_3857 = self.village.plots.to_crs('EPSG:3857')

    def generate_candidates_grid(self, 
                                  global_dx: float, 
                                  global_dy: float,
                                  dx_range: tuple[float, float, float] = (-20.0, 20.0, 2.0),
                                  dy_range: tuple[float, float, float] = (-20.0, 20.0, 2.0),
                                  theta_range: tuple[float, float, float] = (-5.0, 5.0, 2.0)
                                 ) -> list[tuple[float, float, float]]:
        """Create a list of candidate transformations (dx, dy, theta) relative to global offset."""
        dxs = np.arange(global_dx + dx_range[0], global_dx + dx_range[1] + 1e-5, dx_range[2])
        dys = np.arange(global_dy + dy_range[0], global_dy + dy_range[1] + 1e-5, dy_range[2])
        thetas = np.arange(theta_range[0], theta_range[1] + 1e-5, theta_range[2])
        
        candidates = []
        for dx in dxs:
            for dy in dys:
                for theta in thetas:
                    candidates.append((float(dx), float(dy), float(theta)))
                    
        # Always add the exact global shift with 0 rotation
        if (global_dx, global_dy, 0.0) not in candidates:
            candidates.append((float(global_dx), float(global_dy), 0.0))
            
        # Always add the original state (0, 0, 0)
        if (0.0, 0.0, 0.0) not in candidates:
            candidates.append((0.0, 0.0, 0.0))
            
        return candidates

    def evaluate_candidates_vectorized(self, 
                                       coords_3857: np.ndarray, 
                                       centroid_3857: tuple[float, float],
                                       candidates: list[tuple[float, float, float]]
                                      ) -> tuple[np.ndarray, np.ndarray]:
        """Compute boundary and imagery edge scores for all candidates using vectorized numpy operations.

        `coords_3857` is shape (N, 2), `centroid_3857` is (xc, yc).
        Returns:
            bnd_scores: array of shape (C,)
            edge_scores: array of shape (C,)
        """
        N = coords_3857.shape[0]
        C = len(candidates)
        xc, yc = centroid_3857
        
        # Extract candidate arrays
        dx = np.array([c[0] for c in candidates], dtype=np.float32)
        dy = np.array([c[1] for c in candidates], dtype=np.float32)
        theta = np.array([c[2] for c in candidates], dtype=np.float32)
        
        # 1. Transform coordinates relative to centroid
        dx_c = coords_3857[:, 0:1] - xc # (N, 1)
        dy_c = coords_3857[:, 1:2] - yc # (N, 1)
        
        # 2. Vectorized rotation
        rad = np.radians(theta) # (C,)
        cos_t = np.cos(rad)[None, :] # (1, C)
        sin_t = np.sin(rad)[None, :] # (1, C)
        
        x_rot = xc + dx_c * cos_t - dy_c * sin_t # (N, C)
        y_rot = yc + dx_c * sin_t + dy_c * cos_t # (N, C)
        
        # 3. Vectorized translation
        x_new = x_rot + dx[None, :] # (N, C)
        y_new = y_rot + dy[None, :] # (N, C)
        
        # 4. Score on Boundaries Layer
        bnd_scores = np.zeros(C, dtype=np.float32)
        if self.scorer.bnd_layer is not None:
            # Map to pixels
            cols, rows = self.scorer.bnd_inv_transform * (x_new, y_new)
            cols = np.round(cols).astype(np.int32)
            rows = np.round(rows).astype(np.int32)
            
            # Check in-bounds
            mask = (cols >= 0) & (cols < self.scorer.bnd_shape[1]) & (rows >= 0) & (rows < self.scorer.bnd_shape[0])
            
            # Flat lookups
            flat_rows = np.clip(rows.flatten(), 0, self.scorer.bnd_shape[0] - 1)
            flat_cols = np.clip(cols.flatten(), 0, self.scorer.bnd_shape[1] - 1)
            flat_vals = self.scorer.bnd_layer[flat_rows, flat_cols]
            
            vals = flat_vals.reshape(N, C)
            vals[~mask] = 0.0
            
            # Sum up and divide by in-bounds count
            in_bounds_count = np.sum(mask, axis=0)
            bnd_scores = np.sum(vals, axis=0) / np.maximum(in_bounds_count, 1)

        # 5. Score on Imagery Edge Layer
        cols, rows = self.scorer.img_inv_transform * (x_new, y_new)
        cols = np.round(cols).astype(np.int32)
        rows = np.round(rows).astype(np.int32)
        
        mask = (cols >= 0) & (cols < self.scorer.img_shape[1]) & (rows >= 0) & (rows < self.scorer.img_shape[0])
        flat_rows = np.clip(rows.flatten(), 0, self.scorer.img_shape[0] - 1)
        flat_cols = np.clip(cols.flatten(), 0, self.scorer.img_shape[1] - 1)
        flat_vals = self.scorer.img_layer[flat_rows, flat_cols]
        
        vals = flat_vals.reshape(N, C)
        vals[~mask] = 0.0
        
        in_bounds_count = np.sum(mask, axis=0)
        edge_scores = np.sum(vals, axis=0) / np.maximum(in_bounds_count, 1)
        
        return bnd_scores, edge_scores
