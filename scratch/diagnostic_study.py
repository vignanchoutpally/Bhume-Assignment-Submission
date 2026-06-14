import sys
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.affinity import translate, rotate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io import load
from src.scoring import VillageScorer
from src.alignment import CandidateSearchEngine, sample_points_on_boundary, find_plot_neighbors
from src.confidence import compute_confidence_legacy, compute_confidence_calibrated
from src.evaluate import score

def run_diagnostic_for_village(village_dir: str) -> dict:
    village = load(village_dir)
    res = abs(VillageScorer(village, sigma_boundary=2.0, sigma_edge=1.0).img_transform[0])
    scorer = VillageScorer(village, sigma_boundary=2.0, sigma_edge=1.0)
    engine = CandidateSearchEngine(scorer)
    
    # 1. Global Offset
    if village.example_truths is not None and "nashik" in str(village_dir):
        # For Nashik, we had ground truths to guide it
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
    else:
        # Auto-estimation (for malatavadi)
        # Fix the plots_3857 sort key error
        plots_3857 = engine.plots_3857.copy()
        plots_3857['temp_area'] = plots_3857.geometry.area
        plots_sorted = plots_3857.sort_values(by='temp_area', ascending=False)
        sample_plots = plots_sorted.head(50)
        
        plot_coords = []
        plot_centroids = []
        for pn in sample_plots.index:
            geom = sample_plots.loc[pn, 'geometry']
            coords = sample_points_on_boundary(geom, step=res)
            plot_coords.append(coords)
            plot_centroids.append((geom.centroid.x, geom.centroid.y))
            
        dxs_search = np.arange(-30.0, 30.1, max(2.0, res * 1.5))
        dys_search = np.arange(-30.0, 30.1, max(2.0, res * 1.5))
        candidates_search = [(dx, dy, 0.0) for dx in dxs_search for dy in dys_search]
        
        candidate_scores = np.zeros(len(candidates_search), dtype=np.float32)
        for coords, centroid in zip(plot_coords, plot_centroids):
            bnd_sc, _ = engine.evaluate_candidates_vectorized(coords, centroid, candidates_search)
            candidate_scores += bnd_sc
            
        best_idx = np.argmax(candidate_scores)
        global_dx, global_dy = float(candidates_search[best_idx][0]), float(candidates_search[best_idx][1])

    # Weights and parameters
    w_boundary = 1.0
    w_edge = 0.2
    w_global = 0.01
    w_smooth = 0.05
    
    candidates_trans = engine.generate_candidates_grid(
        global_dx, global_dy, 
        dx_range=(-20.0, 20.0, 1.0), 
        dy_range=(-20.0, 20.0, 1.0), 
        theta_range=(0.0, 0.0, 1.0)
    )
    
    neighbors_map = find_plot_neighbors(engine.plots_utm)
    
    plot_dx = {}
    plot_dy = {}
    plot_theta = {}
    sampled_coords = {}
    centroids_3857 = {}
    trans_scores_cache = {}
    
    sample_step = max(1.0, res)
    for pn in engine.plots_3857.index:
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
        
    for iteration in range(2):
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
            plot_dx[pn] = best_c[0]
            plot_dy[pn] = best_c[1]
            
    # Fine Rotation Search
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
        
    # Gather plot data
    plot_data = {}
    for pn in engine.plots_3857.index:
        dx_opt = plot_dx[pn]
        dy_opt = plot_dy[pn]
        theta_opt = plot_theta[pn]
        
        # Calculate neighbors list
        neighbors = neighbors_map.get(pn, [])
        neigh_dx = np.mean([plot_dx[n] for n in neighbors]) if neighbors else global_dx
        neigh_dy = np.mean([plot_dy[n] for n in neighbors]) if neighbors else global_dy
        neighbor_deviation = float(np.sqrt((dx_opt - neigh_dx)**2 + (dy_opt - neigh_dy)**2))
        
        recorded = engine.plots_utm.loc[pn, 'recorded_area_sqm']
        pot_kharaba_sqm = engine.plots_utm.loc[pn, 'pot_kharaba_ha']
        pot_kharaba_sqm = pot_kharaba_sqm * 10000 if not np.isnan(pot_kharaba_sqm) else 0.0
        total_recorded = (recorded + pot_kharaba_sqm) if not np.isnan(recorded) else None
        drawn_area = engine.plots_utm.loc[pn, 'geometry'].area
        area_ratio = (drawn_area / total_recorded) if total_recorded is not None and total_recorded > 0 else None
        
        margin_candidates = [(float(dx), float(dy), 0.0) for dx in np.arange(dx_opt - 10, dx_opt + 10.1, 2) 
                             for dy in np.arange(dy_opt - 10, dy_opt + 10.1, 2)]
        if (0.0, 0.0, 0.0) not in margin_candidates:
            margin_candidates.append((0.0, 0.0, 0.0))
            
        coords = sampled_coords[pn]
        centroid = centroids_3857[pn]
        bnd_scores, edge_scores = engine.evaluate_candidates_vectorized(coords, centroid, margin_candidates)
        scores = w_boundary * bnd_scores + w_edge * edge_scores
        
        orig_idx = margin_candidates.index((0.0, 0.0, 0.0))
        bnd_orig = float(bnd_scores[orig_idx])
        edge_orig = float(edge_scores[orig_idx])
        
        best_idx = np.argmax(scores)
        bnd_best = float(bnd_scores[best_idx])
        edge_best = float(edge_scores[best_idx])
        
        # Penalties
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
        
        def get_second_best_score(scores, candidates, best_c):
            bx, by, bt = best_c
            second_best = 0.0
            for s, c in zip(scores, candidates):
                cx, cy, ct = c
                dist = np.sqrt((cx - bx)**2 + (cy - by)**2)
                if dist >= 2.0:
                    if s > second_best:
                        second_best = s
            return second_best
            
        second_best_score = get_second_best_score(scores, margin_candidates, best_c_actual)
        
        conf_legacy = compute_confidence_legacy(
            best_score=best_score,
            original_score=original_score,
            second_best_score=second_best_score,
            best_boundary_score=best_bnd_score,
            best_edge_score=best_edge_score,
            area_ratio=area_ratio,
            neighbor_deviation=neighbor_deviation,
            w_boundary=w_boundary,
            w_edge=w_edge
        )
        
        conf_calib, signals = compute_confidence_calibrated(
            best_boundary_score=best_bnd_score,
            best_edge_score=best_edge_score,
            area_ratio=area_ratio,
            best_score=best_score,
            second_best_score=second_best_score,
            neighbor_deviation=neighbor_deviation
        )
        
        # Raw Evidence scores
        orig_evidence = w_boundary * bnd_orig + w_edge * edge_orig
        corrected_evidence = w_boundary * bnd_best + w_edge * edge_best
        evidence_imp = corrected_evidence - orig_evidence
        
        # Let's save standard flagging rules decisions
        is_geom_flagged = False
        flag_reason = ""
        if area_ratio is not None and abs(area_ratio - 1.0) > 0.20:
            is_geom_flagged = True
            flag_reason = "area ratio"
        elif best_score - original_score < 0.03:
            is_geom_flagged = True
            flag_reason = "insufficient improvement"
        elif best_score - second_best_score < 0.001:
            is_geom_flagged = True
            flag_reason = "alignment ambiguous"
        elif conf_legacy < 0.40:
            is_geom_flagged = True
            flag_reason = "low legacy conf"
            
        plot_data[pn] = {
            "dx": dx_opt,
            "dy": dy_opt,
            "theta": theta_opt,
            "shift_dist": float(np.sqrt(dx_opt**2 + dy_opt**2)),
            "conf_legacy": conf_legacy,
            "conf_calib": conf_calib,
            "orig_evidence": orig_evidence,
            "corrected_evidence": corrected_evidence,
            "evidence_improvement": evidence_imp,
            "is_geom_flagged": is_geom_flagged,
            "flag_reason": flag_reason,
            "area_ratio": area_ratio,
            "neighbor_deviation": neighbor_deviation
        }
        
    # Post-optimization overlap check simulation
    final_geoms_3857 = {}
    for pn in engine.plots_3857.index:
        geom_original = engine.plots_3857.loc[pn, 'geometry']
        if not plot_data[pn]["is_geom_flagged"]:
            g_shifted = translate(geom_original, plot_data[pn]["dx"], plot_data[pn]["dy"])
            g_shifted = rotate(g_shifted, plot_data[pn]["theta"], origin='centroid')
            final_geoms_3857[pn] = g_shifted
        else:
            final_geoms_3857[pn] = geom_original
            
    for pn in engine.plots_3857.index:
        if plot_data[pn]["is_geom_flagged"]:
            continue
        geom_shifted = final_geoms_3857[pn]
        neighbors = neighbors_map.get(pn, [])
        for n in neighbors:
            n_geom = final_geoms_3857[n]
            if geom_shifted.intersects(n_geom):
                inter_area = geom_shifted.intersection(n_geom).area
                overlap_frac = inter_area / geom_shifted.area
                if overlap_frac > 0.10:
                    plot_data[pn]["is_geom_flagged"] = True
                    plot_data[pn]["flag_reason"] = "overlap violation"
                    break

    return {
        "village": village,
        "engine": engine,
        "plot_data": plot_data,
        "neighbors_map": neighbors_map,
        "global_shift": (global_dx, global_dy)
    }

# Run diagnostics
print("Running diagnostic loading...")
v1_res = run_diagnostic_for_village("data/34855_vadnerbhairav_chandavad_nashik")
v2_res = run_diagnostic_for_village("data/malatavadi")

# Let's print comparisons
v1_data = v1_res["plot_data"]
v2_data = v2_res["plot_data"]

def print_village_stats(name, data):
    shifts = [d["shift_dist"] for d in data.values()]
    corrected_shifts = [d["shift_dist"] for d in data.values() if not d["is_geom_flagged"]]
    confs = [d["conf_calib"] for d in data.values() if not d["is_geom_flagged"]]
    improvements = [d["evidence_improvement"] for d in data.values() if not d["is_geom_flagged"]]
    
    total = len(data)
    corrected_count = len(confs)
    flagged_count = total - corrected_count
    
    large_shifts = sum(s > 15.0 for s in corrected_shifts)
    large_shifts_pct = large_shifts / max(1, corrected_count) * 100
    
    print(f"\n=== {name} ===")
    print(f"Total plots: {total}")
    print(f"Accepted corrections: {corrected_count} ({corrected_count/total*100:.2f}%)")
    print(f"Flagged plots: {flagged_count} ({flagged_count/total*100:.2f}%)")
    print(f"Average shift (all plots): {np.mean(shifts):.2f}m")
    print(f"Average shift (corrected): {np.mean(corrected_shifts):.2f}m" if corrected_shifts else "Average shift (corrected): N/A")
    print(f"Median shift (corrected): {np.median(corrected_shifts):.2f}m" if corrected_shifts else "Median shift (corrected): N/A")
    print(f"Large shifts (>15m) in corrected: {large_shifts} ({large_shifts_pct:.2f}%)")
    if confs:
        print(f"Confidence range (corrected): [{min(confs):.4f}, {max(confs):.4f}]")
        print(f"Mean confidence (corrected): {np.mean(confs):.4f}")
        print(f"Mean evidence improvement: {np.mean(improvements):.4f}")
    else:
        print("No corrected plots.")

print_village_stats("Vadnerbhairav (Nashik)", v1_data)
print_village_stats("Malatavadi", v2_data)

# Save distributions
# Let's save a detailed report of this comparative diagnostic study
report_lines = []
report_lines.append("# Comparative Diagnostic Study: Vadnerbhairav vs Malatavadi")
report_lines.append("\n## Comparative Table")
report_lines.append("\n| Metric | Vadnerbhairav | Malatavadi |")
report_lines.append("| :--- | :---: | :---: |")

def get_row(label, key, is_mean=True, corrected_only=True):
    v1_vals = [d[key] for d in v1_data.values() if not d["is_geom_flagged"]] if corrected_only else [d[key] for d in v1_data.values()]
    v2_vals = [d[key] for d in v2_data.values() if not d["is_geom_flagged"]] if corrected_only else [d[key] for d in v2_data.values()]
    
    v1_res = np.mean(v1_vals) if is_mean else np.median(v1_vals)
    v2_res = np.mean(v2_vals) if is_mean else np.median(v2_vals)
    return f"| {label} | {v1_res:.4f} | {v2_res:.4f} |"

report_lines.append(f"| Total plots | {len(v1_data)} | {len(v2_data)} |")
report_lines.append(f"| Corrections Accepted | {sum(not d['is_geom_flagged'] for d in v1_data.values())} ({sum(not d['is_geom_flagged'] for d in v1_data.values())/len(v1_data)*100:.1f}%) | {sum(not d['is_geom_flagged'] for d in v2_data.values())} ({sum(not d['is_geom_flagged'] for d in v2_data.values())/len(v2_data)*100:.1f}%) |")
report_lines.append(get_row("Average Shift (Corrected)", "shift_dist", is_mean=True, corrected_only=True))
report_lines.append(get_row("Median Shift (Corrected)", "shift_dist", is_mean=False, corrected_only=True))
report_lines.append(get_row("Mean Calibrated Confidence", "conf_calib", is_mean=True, corrected_only=True))
report_lines.append(get_row("Mean Evidence Improvement", "evidence_improvement", is_mean=True, corrected_only=True))

v1_large = sum(d["shift_dist"] > 15.0 for d in v1_data.values() if not d["is_geom_flagged"])
v2_large = sum(d["shift_dist"] > 15.0 for d in v2_data.values() if not d["is_geom_flagged"])
v1_corr = sum(not d['is_geom_flagged'] for d in v1_data.values())
v2_corr = sum(not d['is_geom_flagged'] for d in v2_data.values())
report_lines.append(f"| Large shifts (>15m) in Corrected | {v1_large} ({v1_large/max(1, v1_corr)*100:.1f}%) | {v2_large} ({v2_large/max(1, v2_corr)*100:.1f}%) |")

print("\nSaved report starter.")
Path("scratch/diagnostic_report.md").write_text("\n".join(report_lines))
