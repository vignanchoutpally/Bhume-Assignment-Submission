"""Confidence model for boundary correction predictions with upper range compression."""

from __future__ import annotations

import numpy as np

def compute_confidence_legacy(
    best_score: float,
    original_score: float,
    second_best_score: float,
    best_boundary_score: float,
    best_edge_score: float,
    area_ratio: float | None,
    neighbor_deviation: float,
    w_boundary: float = 0.7,
    w_edge: float = 0.3
) -> float:
    """Calculate the legacy confidence score used to preserve the exact geometric flagging decisions."""
    if best_boundary_score < 0.05:
        w_bnd_eff = 0.2
        w_edge_eff = 0.8
        norm_val = 0.20
    else:
        w_bnd_eff = 0.7
        w_edge_eff = 0.3
        norm_val = 0.25
        
    raw_align_strength = w_bnd_eff * best_boundary_score + w_edge_eff * best_edge_score
    base_conf = min(1.0, raw_align_strength / norm_val)
    
    improvement = best_score - original_score
    if improvement <= 0.0:
        improvement_scale = 0.0
    elif improvement < 0.05:
        improvement_scale = 0.3 + 0.7 * (improvement / 0.05)
    else:
        improvement_scale = 1.0
        
    margin = best_score - second_best_score
    if margin <= 0.0:
        margin_scale = 0.0
    elif margin < 0.005:
        margin_scale = 0.6 + 0.4 * (margin / 0.005)
    else:
        margin_scale = 1.0
        
    if neighbor_deviation <= 2.0:
        neighbor_scale = 1.0
    elif neighbor_deviation < 8.0:
        neighbor_scale = 1.0 - (neighbor_deviation - 2.0) * (0.8 / 6.0)
    else:
        neighbor_scale = 0.2
        
    if area_ratio is not None:
        dev = abs(area_ratio - 1.0)
        if dev <= 0.05:
            area_scale = 1.0
        elif dev < 0.20:
            area_scale = 1.0 - (dev - 0.05) * (1.0 / 0.15)
        else:
            area_scale = 0.0
    else:
        area_scale = 1.0
        
    confidence = base_conf * improvement_scale * margin_scale * neighbor_scale * area_scale
    if confidence > 0.90:
        confidence = 0.90 + 0.02 * (confidence - 0.90) / 0.10
        
    return float(np.clip(confidence, 0.0, 1.0))


def compute_confidence_calibrated(
    best_boundary_score: float,
    best_edge_score: float,
    area_ratio: float | None,
    best_score: float,
    second_best_score: float,
    neighbor_deviation: float
) -> tuple[float, dict[str, float]]:
    """Compute a calibrated confidence score strictly as a monotonic combination of 5 signals.
    
    Returns:
        (confidence, signals_dict)
    """
    # 1. Alignment score (clamped boundary score)
    s_align = min(1.0, max(0.0, best_boundary_score / 0.35))
    
    # 2. Edge evidence score (clamped edge score)
    s_edge = min(1.0, max(0.0, best_edge_score / 0.60))
    
    # 3. Shape plausibility score (area ratio deviation)
    if area_ratio is not None:
        s_shape = 1.0 - min(1.0, max(0.0, abs(area_ratio - 1.0) / 0.20))
    else:
        s_shape = 1.0
        
    # 4. Ambiguity score (best vs second-best margin)
    s_ambig = min(1.0, max(0.0, (best_score - second_best_score) / 0.08))
    
    # 5. Neighborhood consistency score (spatial shift consistency)
    s_neigh = 1.0 - min(1.0, max(0.0, neighbor_deviation / 15.0))
    
    # Monotonic combination weights
    w_align = 0.0769
    w_edge = 0.0769
    w_shape = 0.6923
    w_ambig = 0.0769
    w_neigh = 0.0769
    
    raw_conf = (
        w_align * s_align +
        w_edge * s_edge +
        w_shape * s_shape +
        w_ambig * s_ambig +
        w_neigh * s_neigh
    ) / (w_align + w_edge + w_shape + w_ambig + w_neigh)
    
    # Soft upper range compression to prevent flat capping and overconfidence (>0.95)
    if raw_conf > 0.80:
        conf = 0.80 + 0.12 * np.tanh((raw_conf - 0.80) / 0.20)
    else:
        conf = raw_conf
        
    conf = float(np.clip(conf, 0.0, 1.0))
    
    signals = {
        "alignment_score": s_align,
        "edge_evidence_score": s_edge,
        "shape_plausibility_score": s_shape,
        "ambiguity_score": s_ambig,
        "neighborhood_consistency_score": s_neigh
    }
    
    return conf, signals


def compute_confidence(
    best_score: float,
    original_score: float,
    second_best_score: float,
    best_boundary_score: float,
    best_edge_score: float,
    area_ratio: float | None,
    neighbor_deviation: float,
    w_boundary: float = 0.7,
    w_edge: float = 0.3
) -> float:
    """Fallback entry point for compute_confidence, delegates to legacy."""
    return compute_confidence_legacy(
        best_score=best_score,
        original_score=original_score,
        second_best_score=second_best_score,
        best_boundary_score=best_boundary_score,
        best_edge_score=best_edge_score,
        area_ratio=area_ratio,
        neighbor_deviation=neighbor_deviation,
        w_boundary=w_boundary,
        w_edge=w_edge
    )

