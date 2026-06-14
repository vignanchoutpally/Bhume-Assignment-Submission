"""A baseline approach using a global median shift estimated from example truths."""

from __future__ import annotations

import sys
import statistics
from pathlib import Path
import geopandas as gpd
from shapely.affinity import translate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io import Village

def _utm_for(geom) -> str:
    lon = geom.centroid.x
    return f'EPSG:{32600 + int((lon + 180) // 6) + 1}'

def global_median_shift(village: Village, confidence: float = 0.5) -> gpd.GeoDataFrame:
    """Estimate a single translation vector from example truths and apply it globally."""
    if village.example_truths is None:
        raise ValueError(f'{village.slug} has no example truths')

    utm = _utm_for(village.example_truths.geometry.iloc[0])
    official_u = village.plots.to_crs(utm)
    truth_u = village.example_truths.to_crs(utm)

    dxs, dys = [], []
    for pn in village.example_truths.index:
        if pn in official_u.index:
            o = official_u.loc[pn, 'geometry'].centroid
            t = truth_u.loc[pn, 'geometry'].centroid
            dxs.append(t.x - o.x)
            dys.append(t.y - o.y)
            
    if not dxs:
        raise ValueError('No overlapping plots between example truths and cadastre')
        
    mdx, mdy = statistics.median(dxs), statistics.median(dys)

    shifted = official_u.copy()
    shifted['geometry'] = shifted.geometry.apply(lambda g: translate(g, mdx, mdy))
    
    preds = shifted.to_crs('EPSG:4326')
    preds['status'] = 'corrected'
    preds['confidence'] = confidence
    preds['method_note'] = f'global median shift dx={mdx:.2f}m dy={mdy:.2f}m'
    
    return preds[['plot_number', 'status', 'confidence', 'method_note', 'geometry']]
