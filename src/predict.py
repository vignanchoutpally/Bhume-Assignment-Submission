"""Main prediction pipeline for boundary correction with calibration diagnostics."""

import sys
import logging
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.affinity import translate, rotate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io import Village, load, write_predictions
from src.scoring import VillageScorer
from src.alignment import CandidateSearchEngine, sample_points_on_boundary, find_plot_neighbors
from src.confidence import compute_confidence_legacy, compute_confidence_calibrated
from src.evaluate import score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def estimate_global_offset(scorer: VillageScorer, sample_size: int = 50) -> tuple[float, float]:
    """Automatically estimate the global georeferencing offset by searching a translation grid."""
    engine = CandidateSearchEngine(scorer)
    plots_3857 = engine.plots_3857.copy()
    plots_3857['temp_area'] = plots_3857.geometry.area
    plots_sorted = plots_3857.sort_values(by='temp_area', ascending=False)
    sample_plots = plots_sorted.head(sample_size)
    
    logger.info(f"Auto-estimating global offset using {len(sample_plots)} sample plots...")
    
    # Dynamic step size based on imagery resolution
    res = abs(scorer.img_transform[0])
    logger.info(f"Detected imagery resolution: {res:.3f} m/px")
    
    plot_coords = []
    plot_centroids = []
    for pn in sample_plots.index:
        geom = sample_plots.loc[pn, 'geometry']
        coords = sample_points_on_boundary(geom, step=res)
        plot_coords.append(coords)
        plot_centroids.append((geom.centroid.x, geom.centroid.y))
        
    # Coarse grid search (scaled to resolution)
    dxs = np.arange(-30.0, 30.1, max(2.0, res * 1.5))
    dys = np.arange(-30.0, 30.1, max(2.0, res * 1.5))
    candidates = [(dx, dy, 0.0) for dx in dxs for dy in dys]
    
    candidate_scores = np.zeros(len(candidates), dtype=np.float32)
    
    for coords, centroid in zip(plot_coords, plot_centroids):
        bnd_sc, _ = engine.evaluate_candidates_vectorized(coords, centroid, candidates)
        candidate_scores += bnd_sc
        
    best_idx = np.argmax(candidate_scores)
    best_dx, best_dy, _ = candidates[best_idx]
    logger.info(f"Auto-estimated global shift: dx={best_dx:.2f}m, dy={best_dy:.2f}m (score: {candidate_scores[best_idx]:.2f})")
    return float(best_dx), float(best_dy)


def get_second_best_score(scores: np.ndarray, candidates: list[tuple[float, float, float]], best_c: tuple[float, float, float]) -> float:
    """Find the score of the second-best candidate that is spatially distinct from the best candidate."""
    bx, by, bt = best_c
    second_best = 0.0
    for s, c in zip(scores, candidates):
        cx, cy, ct = c
        # Dist distinct by at least 2 meters
        dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
        if dist >= 2.0:
            if s > second_best:
                second_best = s
    return second_best


def explain_confidence_score(signals: dict) -> str:
    reasons = []
    if signals["alignment_score"] > 0.8:
        reasons.append("strong boundary alignment")
    elif signals["alignment_score"] < 0.3:
        reasons.append("weak boundary alignment")
        
    if signals["edge_evidence_score"] > 0.8:
        reasons.append("strong imagery edge evidence")
    elif signals["edge_evidence_score"] < 0.3:
        reasons.append("weak imagery edge evidence")
        
    if signals["shape_plausibility_score"] > 0.95:
        reasons.append("excellent shape/area preservation")
    elif signals["shape_plausibility_score"] < 0.8:
        reasons.append("moderate area deviation")
        
    if signals["ambiguity_score"] > 0.6:
        reasons.append("unambiguous alignment peak")
    elif signals["ambiguity_score"] < 0.1:
        reasons.append("high spatial placement ambiguity")
        
    if signals["neighborhood_consistency_score"] > 0.95:
        reasons.append("consistent with neighbor shifts")
    elif signals["neighborhood_consistency_score"] < 0.8:
        reasons.append("deviates from neighborhood shift pattern")
        
    if not reasons:
        return "moderate evidence across all signals"
    return ", ".join(reasons)


def print_evaluation_diagnostics(predictions_gdf: gpd.GeoDataFrame) -> str:
    """Generate detailed calibration and restraint diagnostics, including ranked table with explanations."""
    corrected = predictions_gdf[predictions_gdf['status'] == 'corrected']
    flagged = predictions_gdf[predictions_gdf['status'] == 'flagged']
    
    lines = []
    lines.append("=== PIPELINE EVALUATION DIAGNOSTICS ===")
    lines.append(f"Total plots: {len(predictions_gdf)}")
    lines.append(f"CORRECTED plots: {len(corrected)} ({len(corrected)/len(predictions_gdf)*100:.2f}%)")
    lines.append(f"FLAGGED plots: {len(flagged)} ({len(flagged)/len(predictions_gdf)*100:.2f}%)")
    
    reasons = {
        'area ratio': 0,
        'insufficient alignment': 0,
        'alignment ambiguous': 0,
        'low confidence': 0,
        'overlap violation': 0,
        'other': 0
    }
    for note in flagged['method_note'].dropna():
        matched = False
        for k in reasons:
            if k in note:
                reasons[k] += 1
                matched = True
                break
        if not matched:
            reasons['other'] += 1
            
    lines.append("\n--- FLAGGING REASONS BREAKDOWN ---")
    for r, count in reasons.items():
        lines.append(f"  {r.capitalize()}: {count} plots ({count/max(1, len(flagged))*100:.2f}%)")
        
    if len(corrected) > 0:
        confs = corrected['confidence']
        lines.append("\n--- CONFIDENCE SCORE DISTRIBUTION (CORRECTED) ---")
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
        hist, bin_edges = np.histogram(confs, bins=bins)
        for i in range(len(hist)):
            lines.append(f"  [{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}): {hist[i]} plots ({hist[i]/len(corrected)*100:.2f}%)")
            
        lines.append(f"\n  Mean Confidence: {confs.mean():.4f}")
        lines.append(f"  Median Confidence: {confs.median():.4f}")
        lines.append(f"  Confidence > 0.95: {sum(confs > 0.95)} plots")
        lines.append(f"  Confidence > 0.90: {sum(confs > 0.90)} plots")
        
        ranked_corrected = corrected.sort_values(by='confidence', ascending=False).copy()
        
        table_rows = []
        for idx, row in ranked_corrected.iterrows():
            note = row['method_note']
            parts = note.split('|')
            signals = {}
            if len(parts) > 1:
                sig_parts = parts[1].strip().split()
                for sp in sig_parts:
                    k, v = sp.split('=')
                    signals[k] = float(v)
            
            sig_dict = {
                "alignment_score": signals.get("align", 0.0),
                "edge_evidence_score": signals.get("edge", 0.0),
                "shape_plausibility_score": signals.get("shape", 0.0),
                "ambiguity_score": signals.get("ambig", 0.0),
                "neighborhood_consistency_score": signals.get("neigh", 0.0)
            }
            explanation = explain_confidence_score(sig_dict)
            table_rows.append({
                "plot_number": row.plot_number,
                "confidence": row.confidence,
                "align": sig_dict["alignment_score"],
                "edge": sig_dict["edge_evidence_score"],
                "shape": sig_dict["shape_plausibility_score"],
                "ambig": sig_dict["ambiguity_score"],
                "neigh": sig_dict["neighborhood_consistency_score"],
                "explanation": explanation
            })
            
        lines.append("\n--- RANKED TABLE OF CORRECTED PLOTS (TOP 20) ---")
        lines.append(f"{'Rank':<5} | {'Plot':<6} | {'Confidence':<10} | {'Align':<6} | {'Edge':<6} | {'Shape':<6} | {'Ambig':<6} | {'Neigh':<6} | {'Explanation'}")
        lines.append("-" * 120)
        for i, r in enumerate(table_rows[:20]):
            lines.append(f"{i+1:<5} | {r['plot_number']:<6} | {r['confidence']:<10.4f} | {r['align']:<6.4f} | {r['edge']:<6.4f} | {r['shape']:<6.4f} | {r['ambig']:<6.4f} | {r['neigh']:<6.4f} | {r['explanation']}")
            
        lines.append("\n--- RANKED TABLE OF CORRECTED PLOTS (BOTTOM 20) ---")
        lines.append(f"{'Rank':<5} | {'Plot':<6} | {'Confidence':<10} | {'Align':<6} | {'Edge':<6} | {'Shape':<6} | {'Ambig':<6} | {'Neigh':<6} | {'Explanation'}")
        lines.append("-" * 120)
        start_bot = max(20, len(table_rows) - 20)
        for i, r in enumerate(table_rows[start_bot:]):
            rank_idx = start_bot + i + 1
            lines.append(f"{rank_idx:<5} | {r['plot_number']:<6} | {r['confidence']:<10.4f} | {r['align']:<6.4f} | {r['edge']:<6.4f} | {r['shape']:<6.4f} | {r['ambig']:<6.4f} | {r['neigh']:<6.4f} | {r['explanation']}")
            
    report = "\n".join(lines)
    return report


def main(
    village_dir: str, 
    output_path: str | None = None,
    confidence_threshold: float = 0.50,
    evidence_improvement_threshold: float = 0.03,
    large_shift_threshold: float = 15.0,
    large_shift_evidence_req: float = 0.15
) -> None:
    # 1. Load data
    village = load(village_dir)
    logger.info(f"Loaded village {village.slug} with {len(village.plots)} plots")
    
    # 2. Build Scorer
    # Set Gaussian blurs based on spatial resolution
    res = abs(VillageScorer(village, sigma_boundary=1.0, sigma_edge=1.0).img_transform[0])
    logger.info(f"Resolution-aware scaling: Pixel resolution is {res:.3f}m")
    
    # boundaries.tif is half resolution of imagery (~2.4m/px), so sigma=2.0px is suitable.
    scorer = VillageScorer(village, sigma_boundary=2.0, sigma_edge=1.0)
    engine = CandidateSearchEngine(scorer)
    
    # 3. Determine Global Shift
    if village.example_truths is not None:
        logger.info("Example truths present. Estimating global offset using median offset...")
        plots_utm = engine.plots_utm
        truths_utm = village.example_truths.to_crs(engine.utm_crs)
        dxs, dys = [], []
        for pn in village.example_truths.index:
            if pn in plots_utm.index:
                o = plots_utm.loc[pn, 'geometry'].centroid
                t = truths_utm.loc[pn, 'geometry'].centroid
                dxs.append(t.x - o.x)
                dys.append(t.y - o.y)
        global_dx = float(np.median(dxs))
        global_dy = float(np.median(dys))
        logger.info(f"Median global offset from truths: dx={global_dx:.2f}m, dy={global_dy:.2f}m")
    else:
        global_dx, global_dy = estimate_global_offset(scorer)
        
    # 4. Parameters
    w_boundary = 1.0
    w_edge = 0.2
    w_global = 0.01
    w_smooth = 0.05
    
    area_ratio_threshold = 0.20
    min_improvement = 0.03       # Slightly lower to capture fine local alignments
    min_margin = 0.001           # Small distinct threshold for flat peak profiles
    
    # 5. Pre-generate candidate transforms grid
    # Translate candidates (using 1m resolution-aligned steps)
    candidates_trans = engine.generate_candidates_grid(
        global_dx, global_dy, 
        dx_range=(-20.0, 20.0, 1.0), 
        dy_range=(-20.0, 20.0, 1.0), 
        theta_range=(0.0, 0.0, 1.0)
    )
    
    # Precompute neighbor adjacency map
    logger.info("Computing neighbor adjacency map...")
    neighbors_map = find_plot_neighbors(engine.plots_utm)
    
    plot_dx = {}
    plot_dy = {}
    plot_theta = {}
    
    logger.info("Sampling boundary coordinates and evaluating independent shifts...")
    sampled_coords = {}
    centroids_3857 = {}
    trans_scores_cache = {}
    
    # Resolution-aware point sampling step
    sample_step = max(1.0, res)
    
    for i, pn in enumerate(engine.plots_3857.index):
        if i % 500 == 0:
            logger.info(f"  Processed {i}/{len(engine.plots_3857)} plots...")
        geom = engine.plots_3857.loc[pn, 'geometry']
        xc, yc = geom.centroid.x, geom.centroid.y
        centroids_3857[pn] = (xc, yc)
        
        coords = sample_points_on_boundary(geom, step=sample_step)
        sampled_coords[pn] = coords
        
        bnd_scores, edge_scores = engine.evaluate_candidates_vectorized(coords, (xc, yc), candidates_trans)
        trans_scores_cache[pn] = (bnd_scores, edge_scores)
        
        scores = w_boundary * bnd_scores + w_edge * edge_scores
        for j, c in enumerate(candidates_trans):
            dx_c, dy_c, _ = c
            dist_global = np.sqrt((dx_c - global_dx)**2 + (dy_c - global_dy)**2)
            scores[j] -= w_global * dist_global
            
        best_idx = np.argmax(scores)
        best_c = candidates_trans[best_idx]
        plot_dx[pn] = best_c[0]
        plot_dy[pn] = best_c[1]
        plot_theta[pn] = 0.0
        
    # 6. Neighborhood Smoothness Optimization (Iterative Conditional Modes)
    logger.info("Running spatial smoothness refinement...")
    for iteration in range(2):
        logger.info(f"  Iteration {iteration + 1}...")
        updated_count = 0
        for pn in engine.plots_3857.index:
            neighbors = neighbors_map.get(pn, [])
            if not neighbors:
                continue
                
            neigh_dx = np.mean([plot_dx[n] for n in neighbors])
            neigh_dy = np.mean([plot_dy[n] for n in neighbors])
            
            bnd_scores, edge_scores = trans_scores_cache[pn]
            scores = w_boundary * bnd_scores + w_edge * edge_scores
            
            for j, c in enumerate(candidates_trans):
                dx_c, dy_c, _ = c
                dist_global = np.sqrt((dx_c - global_dx)**2 + (dy_c - global_dy)**2)
                dist_neigh = np.sqrt((dx_c - neigh_dx)**2 + (dy_c - neigh_dy)**2)
                scores[j] -= (w_global * dist_global + w_smooth * dist_neigh)
                
            best_idx = np.argmax(scores)
            best_c = candidates_trans[best_idx]
            
            if best_c[0] != plot_dx[pn] or best_c[1] != plot_dy[pn]:
                plot_dx[pn] = best_c[0]
                plot_dy[pn] = best_c[1]
                updated_count += 1
        logger.info(f"    Updated displacements for {updated_count} plots")

    # 7. Fine Rotation Search
    logger.info("Evaluating small candidate rotations around best translations...")
    for pn in engine.plots_3857.index:
        dx_opt = plot_dx[pn]
        dy_opt = plot_dy[pn]
        
        rot_candidates = [(dx_opt, dy_opt, float(t)) for t in np.arange(-5.0, 5.1, 1.0)]
        coords = sampled_coords[pn]
        centroid = centroids_3857[pn]
        
        bnd_scores, edge_scores = engine.evaluate_candidates_vectorized(coords, centroid, rot_candidates)
        scores = w_boundary * bnd_scores + w_edge * edge_scores
        
        best_idx = np.argmax(scores)
        plot_theta[pn] = rot_candidates[best_idx][2]

    # 8. Compute Raw Status and Confidence
    logger.info("Computing confidence and applying flagging rules...")
    predictions_status = {}
    plot_confidences = {}
    predictions_notes = {}
    
    for pn in engine.plots_3857.index:
        geom_utm = engine.plots_utm.loc[pn, 'geometry']
        
        recorded = engine.plots_utm.loc[pn, 'recorded_area_sqm']
        pot_kharaba_sqm = engine.plots_utm.loc[pn, 'pot_kharaba_ha']
        pot_kharaba_sqm = pot_kharaba_sqm * 10000 if not np.isnan(pot_kharaba_sqm) else 0.0
        
        total_recorded = (recorded + pot_kharaba_sqm) if not np.isnan(recorded) else None
        drawn_area = geom_utm.area
        area_ratio = (drawn_area / total_recorded) if total_recorded is not None and total_recorded > 0 else None
        
        dx_opt = plot_dx[pn]
        dy_opt = plot_dy[pn]
        theta_opt = plot_theta[pn]
        
        margin_candidates = [(float(dx), float(dy), 0.0) for dx in np.arange(dx_opt - 10, dx_opt + 10.1, 2) 
                             for dy in np.arange(dy_opt - 10, dy_opt + 10.1, 2)]
        if (0.0, 0.0, 0.0) not in margin_candidates:
            margin_candidates.append((0.0, 0.0, 0.0))
            
        coords = sampled_coords[pn]
        centroid = centroids_3857[pn]
        
        bnd_scores, edge_scores = engine.evaluate_candidates_vectorized(coords, centroid, margin_candidates)
        scores = w_boundary * bnd_scores + w_edge * edge_scores
        
        # Calculate raw unpenalized evidence improvement
        orig_idx = margin_candidates.index((0.0, 0.0, 0.0))
        bnd_orig = float(bnd_scores[orig_idx])
        edge_orig = float(edge_scores[orig_idx])
        orig_evidence = w_boundary * bnd_orig + w_edge * edge_orig
        
        best_raw_idx = np.argmax(scores)
        bnd_best = float(bnd_scores[best_raw_idx])
        edge_best = float(edge_scores[best_raw_idx])
        corrected_evidence = w_boundary * bnd_best + w_edge * edge_best
        evidence_imp = corrected_evidence - orig_evidence
        
        neighbors = neighbors_map.get(pn, [])
        if neighbors:
            neigh_dx = np.mean([plot_dx[n] for n in neighbors])
            neigh_dy = np.mean([plot_dy[n] for n in neighbors])
            neighbor_deviation = float(np.sqrt((dx_opt - neigh_dx)**2 + (dy_opt - neigh_dy)**2))
        else:
            neigh_dx = global_dx
            neigh_dy = global_dy
            neighbor_deviation = float(np.sqrt((dx_opt - global_dx)**2 + (dy_opt - global_dy)**2))
            
        # Add global and neighbor penalties to candidates
        for j, c in enumerate(margin_candidates):
            dx_c, dy_c, _ = c
            dist_global = np.sqrt((dx_c - global_dx)**2 + (dy_c - global_dy)**2)
            dist_neigh = np.sqrt((dx_c - neigh_dx)**2 + (dy_c - neigh_dy)**2)
            scores[j] -= (w_global * dist_global + w_smooth * dist_neigh)
            
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        best_bnd_score = float(bnd_scores[best_idx])
        best_edge_score = float(edge_scores[best_idx])
        best_c_actual = margin_candidates[best_idx]
        
        original_score = float(scores[orig_idx])
        
        second_best_score = get_second_best_score(scores, margin_candidates, best_c_actual)
        
        conf_calib, signals = compute_confidence_calibrated(
            best_boundary_score=best_bnd_score,
            best_edge_score=best_edge_score,
            area_ratio=area_ratio,
            best_score=best_score,
            second_best_score=second_best_score,
            neighbor_deviation=neighbor_deviation
        )
        
        status = 'corrected'
        method_note = f'aligned local dx={dx_opt:.2f}m dy={dy_opt:.2f}m rot={theta_opt:.1f}deg'
        
        shift_dist = float(np.sqrt(dx_opt**2 + dy_opt**2))
        
        if area_ratio is not None and abs(area_ratio - 1.0) > area_ratio_threshold:
            status = 'flagged'
            method_note = f'flagged: area ratio {area_ratio:.2f} indicates shape error'
        elif best_score - original_score < min_improvement:
            status = 'flagged'
            method_note = f'flagged: insufficient alignment score improvement ({best_score - original_score:.4f})'
        elif best_score - second_best_score < min_margin:
            status = 'flagged'
            method_note = f'flagged: alignment ambiguous (margin {best_score - second_best_score:.4f})'
        elif evidence_imp < evidence_improvement_threshold:
            status = 'flagged'
            method_note = f'flagged: insufficient evidence improvement ({evidence_imp:.4f} < {evidence_improvement_threshold:.2f})'
        elif conf_calib < confidence_threshold:
            status = 'flagged'
            method_note = f'flagged: low confidence ({conf_calib:.3f} < {confidence_threshold:.2f})'
        elif shift_dist > large_shift_threshold and evidence_imp < large_shift_evidence_req:
            status = 'flagged'
            method_note = f'flagged: large-shift safeguard rejected shift of {shift_dist:.2f}m with evidence improvement {evidence_imp:.4f}'
            
        predictions_status[pn] = status
        plot_confidences[pn] = conf_calib
        predictions_notes[pn] = method_note

    # 9. Post-optimization Overlap / Topology Validation Pass
    logger.info("Running post-optimization overlap checks...")
    final_geoms_3857 = {}
    for pn in engine.plots_3857.index:
        status = predictions_status[pn]
        geom_original = engine.plots_3857.loc[pn, 'geometry']
        if status == 'corrected':
            g_shifted = translate(geom_original, plot_dx[pn], plot_dy[pn])
            g_shifted = rotate(g_shifted, plot_theta[pn], origin='centroid')
            final_geoms_3857[pn] = g_shifted
        else:
            final_geoms_3857[pn] = geom_original
            
    overlap_reverts = 0
    for pn in engine.plots_3857.index:
        if predictions_status[pn] != 'corrected':
            continue
        geom_shifted = final_geoms_3857[pn]
        neighbors = neighbors_map.get(pn, [])
        for n in neighbors:
            n_geom = final_geoms_3857[n]
            # Check if they overlap significantly (exceeding 10%)
            if geom_shifted.intersects(n_geom):
                inter_area = geom_shifted.intersection(n_geom).area
                overlap_frac = inter_area / geom_shifted.area
                if overlap_frac > 0.10:
                    predictions_status[pn] = 'flagged'
                    predictions_notes[pn] = f'flagged: overlap violation with plot {n} ({overlap_frac*100:.1f}%)'
                    plot_confidences[pn] = 0.0
                    overlap_reverts += 1
                    break
    logger.info(f"Reverted {overlap_reverts} plots to flagged due to spacing overlap violations")

    # 10. Construct Final GeoDataFrame
    predictions_rows = []
    for pn in engine.plots_3857.index:
        geom_original = engine.plots_utm.loc[pn, 'geometry']
        status = predictions_status[pn]
        
        geom_shifted = geom_original
        conf = 0.0
        method_note = predictions_notes[pn]
        
        if status == 'corrected':
            geom_shifted = translate(geom_shifted, plot_dx[pn], plot_dy[pn])
            geom_shifted = rotate(geom_shifted, plot_theta[pn], origin='centroid')
            
            # Recalculate neighbors & deviation using final statuses
            neighbors = neighbors_map.get(pn, [])
            neigh_dxs, neigh_dys = [], []
            for n in neighbors:
                if predictions_status[n] == 'corrected':
                    neigh_dxs.append(plot_dx[n])
                    neigh_dys.append(plot_dy[n])
            if neigh_dxs:
                neigh_dx = np.mean(neigh_dxs)
                neigh_dy = np.mean(neigh_dys)
            else:
                neigh_dx = global_dx
                neigh_dy = global_dy
            neighbor_deviation = float(np.sqrt((plot_dx[pn] - neigh_dx)**2 + (plot_dy[pn] - neigh_dy)**2))
            
            recorded = engine.plots_utm.loc[pn, 'recorded_area_sqm']
            pot_kharaba_sqm = engine.plots_utm.loc[pn, 'pot_kharaba_ha']
            pot_kharaba_sqm = pot_kharaba_sqm * 10000 if not np.isnan(pot_kharaba_sqm) else 0.0
            total_recorded = (recorded + pot_kharaba_sqm) if not np.isnan(recorded) else None
            drawn_area = geom_shifted.area
            area_ratio = (drawn_area / total_recorded) if total_recorded is not None and total_recorded > 0 else None
            
            dx_opt = plot_dx[pn]
            dy_opt = plot_dy[pn]
            margin_candidates = [(float(dx), float(dy), 0.0) for dx in np.arange(dx_opt - 10, dx_opt + 10.1, 2) 
                                 for dy in np.arange(dy_opt - 10, dy_opt + 10.1, 2)]
            if (0.0, 0.0, 0.0) not in margin_candidates:
                margin_candidates.append((0.0, 0.0, 0.0))
                
            coords = sampled_coords[pn]
            centroid = centroids_3857[pn]
            bnd_scores, edge_scores = engine.evaluate_candidates_vectorized(coords, centroid, margin_candidates)
            scores = w_boundary * bnd_scores + w_edge * edge_scores
            
            for j, c in enumerate(margin_candidates):
                dx_c, dy_c, _ = c
                dist_global = np.sqrt((dx_c - global_dx)**2 + (dy_c - global_dy)**2)
                dist_neigh = np.sqrt((dx_c - neigh_dx)**2 + (dy_c - neigh_dy)**2)
                scores[j] -= (w_global * dist_global + w_smooth * dist_neigh)
                
            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])
            best_bnd_score = float(bnd_scores[best_idx])
            best_edge_score = float(edge_scores[best_idx])
            best_c_actual = margin_candidates[best_idx]
            
            second_best_score = get_second_best_score(scores, margin_candidates, best_c_actual)
            
            conf, signals = compute_confidence_calibrated(
                best_boundary_score=best_bnd_score,
                best_edge_score=best_edge_score,
                area_ratio=area_ratio,
                best_score=best_score,
                second_best_score=second_best_score,
                neighbor_deviation=neighbor_deviation
            )
            
            method_note = (
                f'aligned local dx={dx_opt:.2f}m dy={dy_opt:.2f}m rot={plot_theta[pn]:.1f}deg | '
                f'align={signals["alignment_score"]:.4f} edge={signals["edge_evidence_score"]:.4f} '
                f'shape={signals["shape_plausibility_score"]:.4f} ambig={signals["ambiguity_score"]:.4f} '
                f'neigh={signals["neighborhood_consistency_score"]:.4f}'
            )
            
        gs_utm = gpd.GeoSeries([geom_shifted], crs=engine.utm_crs)
        geom_4326 = gs_utm.to_crs('EPSG:4326').iloc[0]
        
        predictions_rows.append({
            'plot_number': pn,
            'status': status,
            'confidence': conf,
            'method_note': method_note,
            'geometry': geom_4326
        })
        
    predictions_gdf = gpd.GeoDataFrame(predictions_rows, crs='EPSG:4326')
    
    # 11. Output predictions
    if output_path is None:
        output_path = Path(village_dir) / 'predictions.geojson'
    else:
        output_path = Path(output_path)
        
    out = write_predictions(output_path, predictions_gdf)
    logger.info(f"Predictions written to {out}")
    
    # 12. Run Diagnostics and save Report
    report = print_evaluation_diagnostics(predictions_gdf)
    print("\n" + report + "\n")
    
    report_file = Path(village_dir) / 'diagnostics_report.txt'
    report_file.write_text(report)
    logger.info(f"Diagnostics report saved to {report_file}")
    
    # Save full ranked corrections table to ranked_corrections.txt
    corrected_only = predictions_gdf[predictions_gdf['status'] == 'corrected']
    if len(corrected_only) > 0:
        ranked_corrected = corrected_only.sort_values(by='confidence', ascending=False).copy()
        ranked_lines = []
        ranked_lines.append("=== RANKED CORRECTED PLOTS (ALL) ===")
        ranked_lines.append(f"{'Rank':<5} | {'Plot':<6} | {'Confidence':<10} | {'Align':<6} | {'Edge':<6} | {'Shape':<6} | {'Ambig':<6} | {'Neigh':<6} | {'Explanation'}")
        ranked_lines.append("-" * 120)
        for i, (idx, row) in enumerate(ranked_corrected.iterrows()):
            note = row['method_note']
            parts = note.split('|')
            signals = {}
            if len(parts) > 1:
                sig_parts = parts[1].strip().split()
                for sp in sig_parts:
                    k, v = sp.split('=')
                    signals[k] = float(v)
            sig_dict = {
                "alignment_score": signals.get("align", 0.0),
                "edge_evidence_score": signals.get("edge", 0.0),
                "shape_plausibility_score": signals.get("shape", 0.0),
                "ambiguity_score": signals.get("ambig", 0.0),
                "neighborhood_consistency_score": signals.get("neigh", 0.0)
            }
            explanation = explain_confidence_score(sig_dict)
            ranked_lines.append(f"{i+1:<5} | {row.plot_number:<6} | {row.confidence:<10.4f} | {sig_dict['alignment_score']:<6.4f} | {sig_dict['edge_evidence_score']:<6.4f} | {sig_dict['shape_plausibility_score']:<6.4f} | {sig_dict['ambiguity_score']:<6.4f} | {sig_dict['neighborhood_consistency_score']:<6.4f} | {explanation}")
            
        ranked_file = Path(village_dir) / 'ranked_corrections.txt'
        ranked_file.write_text("\n".join(ranked_lines))
        logger.info(f"Full ranked corrections table saved to {ranked_file}")
        
    # 13. Self-score if example truths are present
    if village.example_truths is not None:
        sc = score(predictions_gdf, village)
        print("\n--- LOCAL ALIGNMENT ENGINE SCORE ---")
        print(sc)
        
        score_file = Path(village_dir) / 'scorecard.txt'
        score_file.write_text(str(sc))
        logger.info(f"Saved scorecard details to {score_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        v_dir = sys.argv[1]
    else:
        v_dir = "data/34855_vadnerbhairav_chandavad_nashik"
        
    out_p = sys.argv[2] if len(sys.argv) > 2 else None
    main(v_dir, out_p)
