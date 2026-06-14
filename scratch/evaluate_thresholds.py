import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
from shapely.affinity import translate, rotate
from diagnostic_study import run_diagnostic_for_village
from src.evaluate import score

# Load the cached diagnostic runs
print("Running diagnostic data load...")
v1_res = run_diagnostic_for_village("data/34855_vadnerbhairav_chandavad_nashik")
v2_res = run_diagnostic_for_village("data/malatavadi")

def evaluate_thresholds(res, conf_threshold, evidence_imp_threshold, large_shift_threshold=15.0, large_shift_evidence_req=0.15):
    village = res["village"]
    engine = res["engine"]
    plot_data = res["plot_data"]
    neighbors_map = res["neighbors_map"]
    global_dx, global_dy = res["global_shift"]
    
    # Simulate flagging based on custom rules
    sim_status = {}
    sim_conf = {}
    
    for pn, d in plot_data.items():
        # Baseline flagging
        is_flagged = d["is_geom_flagged"]
        
        # Override flagging with our new rules if not already flagged by area ratio or ambiguity
        if not is_flagged:
            # Task 4: Accept if corrected evidence significantly exceeds original evidence score
            # and confidence exceeds threshold
            evidence_ok = (d["evidence_improvement"] >= evidence_imp_threshold)
            conf_ok = (d["conf_calib"] >= conf_threshold)
            
            # Task 6: Large-shift safeguard (reject if shift > 15m unless evidence is exceptionally strong)
            shift_ok = True
            if d["shift_dist"] > large_shift_threshold:
                if d["evidence_improvement"] < large_shift_evidence_req:
                    shift_ok = False
                    
            if not (evidence_ok and conf_ok and shift_ok):
                is_flagged = True
                
        sim_status[pn] = 'flagged' if is_flagged else 'corrected'
        sim_conf[pn] = 0.0 if is_flagged else d["conf_calib"]
        
    # Overlap pass simulation
    final_geoms_3857 = {}
    for pn in engine.plots_3857.index:
        geom_original = engine.plots_3857.loc[pn, 'geometry']
        if sim_status[pn] == 'corrected':
            d = plot_data[pn]
            g_shifted = translate(geom_original, d["dx"], d["dy"])
            g_shifted = rotate(g_shifted, d["theta"], origin='centroid')
            final_geoms_3857[pn] = g_shifted
        else:
            final_geoms_3857[pn] = geom_original
            
    for pn in engine.plots_3857.index:
        if sim_status[pn] != 'corrected':
            continue
        geom_shifted = final_geoms_3857[pn]
        neighbors = neighbors_map.get(pn, [])
        for n in neighbors:
            n_geom = final_geoms_3857[n]
            if geom_shifted.intersects(n_geom):
                inter_area = geom_shifted.intersection(n_geom).area
                overlap_frac = inter_area / geom_shifted.area
                if overlap_frac > 0.10:
                    sim_status[pn] = 'flagged'
                    sim_conf[pn] = 0.0
                    break
                    
    # Construct GeoDataFrame to pass to evaluate.score
    predictions_rows = []
    for pn in engine.plots_3857.index:
        geom_original = engine.plots_utm.loc[pn, 'geometry']
        status = sim_status[pn]
        geom_shifted = geom_original
        if status == 'corrected':
            d = plot_data[pn]
            geom_shifted = translate(geom_shifted, d["dx"], d["dy"])
            geom_shifted = rotate(geom_shifted, d["theta"], origin='centroid')
        gs_utm = gpd.GeoSeries([geom_shifted], crs=engine.utm_crs)
        geom_4326 = gs_utm.to_crs('EPSG:4326').iloc[0]
        
        predictions_rows.append({
            'plot_number': pn,
            'status': status,
            'confidence': sim_conf[pn],
            'geometry': geom_4326
        })
        
    predictions_gdf = gpd.GeoDataFrame(predictions_rows, crs='EPSG:4326')
    sc = score(predictions_gdf, village)
    return sc, sum(s == 'corrected' for s in sim_status.values())

print("\n=== Nashik Threshold Matrix ===")
print(f"{'Conf T':<8} | {'Ev Imp T':<10} | {'Corr %':<8} | {'Med IoU':<8} | {'Improve':<8} | {'Spearman':<8}")
print("-" * 65)

def f_val(v, fmt_str="{:.3f}"):
    return "N/A" if v is None else fmt_str.format(v)

# Grid search for Nashik
for ct in [0.40, 0.45, 0.50, 0.55]:
    for et in [0.02, 0.04, 0.06, 0.08]:
        sc, n_corr = evaluate_thresholds(v1_res, ct, et)
        corr_pct = n_corr / len(v1_res["plot_data"]) * 100
        mi = f_val(sc.median_iou_pred)
        mimp = f_val(sc.median_improvement, "{:+0.3f}")
        sp = f_val(sc.spearman_conf_vs_iou)
        print(f"{ct:<8.2f} | {et:<10.2f} | {corr_pct:<7.1f}% | {mi:<8} | {mimp:<8} | {sp:<8}")

print("\n=== Malatavadi Threshold Matrix ===")
print(f"{'Conf T':<8} | {'Ev Imp T':<10} | {'Corr %':<8} | {'Med IoU':<8} | {'Improve':<8} | {'Spearman':<8}")
print("-" * 65)

# Grid search for Malatavadi
for ct in [0.40, 0.45, 0.50, 0.55]:
    for et in [0.02, 0.04, 0.06, 0.08]:
        sc, n_corr = evaluate_thresholds(v2_res, ct, et)
        corr_pct = n_corr / len(v2_res["plot_data"]) * 100
        mi = f_val(sc.median_iou_pred)
        mimp = f_val(sc.median_improvement, "{:+0.3f}")
        sp = f_val(sc.spearman_conf_vs_iou)
        print(f"{ct:<8.2f} | {et:<10.2f} | {corr_pct:<7.1f}% | {mi:<8} | {mimp:<8} | {sp:<8}")
