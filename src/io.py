"""Loading a village bundle, handling CRS transitions, and coordinate projections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

logger = logging.getLogger(__name__)

@dataclass
class Patch:
    """An image crop around a plot.

    `image` is (H, W, C) numpy array. `transform` maps pixel (col, row) → imagery-CRS (x, y);
    `bounds` is (left, bottom, right, top) in the imagery CRS.
    """
    image: np.ndarray
    transform: object
    crs: str
    bounds: tuple[float, float, float, float]


@dataclass
class Village:
    """One village bundle, loaded and CRS-sorted.

    `plots` is the official (shifted) cadastre — a GeoDataFrame in EPSG:4326.
    `example_truths` is the public sample of hand-aligned boundaries.
    """
    slug: str
    dir: Path
    plots: gpd.GeoDataFrame
    imagery_path: Path
    boundaries_path: Path | None
    example_truths: gpd.GeoDataFrame | None

    def plot(self, plot_number: str):
        """The official geometry for one plot."""
        return self.plots.loc[str(plot_number), 'geometry']


def load(village_dir: str | Path) -> Village:
    """Load a village bundle from a folder."""
    d = Path(village_dir)
    input_path = d / 'input.geojson'
    imagery_path = d / 'imagery.tif'
    if not input_path.exists():
        raise FileNotFoundError(f'{input_path} not found.')
    if not imagery_path.exists():
        raise FileNotFoundError(f'{imagery_path} not found.')

    plots = gpd.read_file(input_path)
    plots['plot_number'] = plots['plot_number'].astype(str)
    plots = plots.set_index('plot_number', drop=False)

    boundaries_path = d / 'boundaries.tif'
    truths_path = d / 'example_truths.geojson'
    example_truths = None
    if truths_path.exists():
        example_truths = gpd.read_file(truths_path)
        example_truths['plot_number'] = example_truths['plot_number'].astype(str)
        example_truths = example_truths.set_index('plot_number', drop=False)

    return Village(
        slug=d.name,
        dir=d,
        plots=plots,
        imagery_path=imagery_path,
        boundaries_path=boundaries_path if boundaries_path.exists() else None,
        example_truths=example_truths,
    )


def write_predictions(path: str | Path, predictions: gpd.GeoDataFrame) -> Path:
    """Write predictions GeoDataFrame to predictions.geojson in EPSG:4326."""
    required = {'plot_number', 'status', 'geometry'}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f'predictions is missing required columns: {sorted(missing)}')

    gdf = predictions.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    else:
        gdf = gdf.to_crs('EPSG:4326')

    keep = [c for c in ('plot_number', 'status', 'confidence', 'method_note', 'geometry') if c in gdf.columns]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(gdf[keep].to_json())
    return out


def open_imagery(path):
    """Open a raster dataset."""
    return rasterio.open(path)


def _to_imagery_crs(src):
    return Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)


def _to_lonlat_crs(src):
    return Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)


def lonlat_to_pixel(src, lon: float, lat: float) -> tuple[int, int]:
    """Map a lon/lat point to (col, row) pixel coordinates in the imagery."""
    x, y = _to_imagery_crs(src).transform(lon, lat)
    row, col = src.index(x, y)
    return int(col), int(row)


def pixel_to_lonlat(src, col: float, row: float) -> tuple[float, float]:
    """Map a (col, row) pixel back to (lon, lat)."""
    x, y = src.xy(row, col)
    lon, lat = _to_lonlat_crs(src).transform(x, y)
    return float(lon), float(lat)


def geom_to_imagery_crs(src, geom_4326: BaseGeometry) -> BaseGeometry:
    """Reproject geometry into imagery CRS."""
    tf = _to_imagery_crs(src)
    return shp_transform(lambda xs, ys, z=None: tf.transform(xs, ys), geom_4326)


def patch_for_plot(src, geom_4326: BaseGeometry, pad_m: float = 25.0) -> Patch:
    """Read the image crop covering a plot (in lon/lat) padded by pad_m metres."""
    g = geom_to_imagery_crs(src, geom_4326)
    minx, miny, maxx, maxy = g.bounds
    left, bottom, right, top = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m

    dl, db, dr, dt = src.bounds
    left, bottom, right, top = max(left, dl), max(bottom, db), min(right, dr), min(top, dt)
    if right <= left or top <= bottom:
        raise ValueError('plot bounding box does not overlap the imagery extent')

    window = from_bounds(left, bottom, right, top, transform=src.transform)
    count = src.count
    if count >= 3:
        rgb = src.read([1, 2, 3], window=window)
        image = np.transpose(rgb, (1, 2, 0))
    else:
        # Single band (e.g. boundaries raster)
        band = src.read(1, window=window)
        image = band
    return Patch(
        image=image,
        transform=src.window_transform(window),
        crs=str(src.crs),
        bounds=(left, bottom, right, top),
    )
