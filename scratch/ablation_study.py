import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np
from shapely.affinity import translate, rotate
from scratch.diagnostic_study import run_diagnostic_for_village
from src.evaluate import score

# Load the cached diagnostic runs
print("Loading diagnostic data...")
v1_res = run_diagnostic_for_village("data/34855_vadnerbhairav_chandavad_nashik")
v2_res = run_diagnostic_for_village("data/malatavadi")

def run_simulation(res, config_name):
    village = res["village"]
    engine = res["engine"]
    plot_data = res["plot_data"]
    neighbors_map = res["neighbors_map"]
    global_dx, global_dy = res["global_shift"]
    
    sim_status = {}
    sim_conf = {}
    
    for pn, d in plot_data.items():
        is_flagged = d["is_geom_flagged"] # area ratio, score improvement < 0.03, margin < 0.001, legacy < 0.40
        
        if config_name == "original":
            # Just keep the baseline flagging
            pass
        elif config_name == "confidence_only":
            if not is_flagged:
                # Flag if calibrated confidence < 0.50
                if d["conf_calib"] < 0.50:
                    is_flagged = True
        elif config_name == "evidence_improvement":
            if not is_flagged:
                # Accept only if confidence >= 0.50 and evidence improvement >= 0.04
                # and shift distance <= 15m (or evidence >= 0.15)
                conf_ok = (d["conf_calib"] >= 0.50)
                evidence_ok = (d["evidence_improvement"] >= 0.04)
                shift_ok = True
                if d["shift_dist"] > 15.0:
                    if d["evidence_improvement"] < 0.15:
                        shift_ok = False
                if not (conf_ok and evidence_ok and shift_ok):
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
                    
    # Construct predictions gdf
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
    n_corrected = sum(s == 'corrected' for s in sim_status.values())
    corr_rate = n_corrected / len(sim_status) * 100
    
    return {
        "corr_rate": f"{corr_rate:.1f}% ({n_corrected}/{len(sim_status)})",
        "median_iou": "N/A" if sc.median_iou_pred is None else f"{sc.median_iou_pred:.3f}",
        "improvement": "N/A" if sc.median_improvement is None else f"{sc.median_improvement:+0.3f}",
        "calibration": "N/A" if sc.spearman_conf_vs_iou is None else f"{sc.spearman_conf_vs_iou:.3f}",
        "centroid_err": "N/A" if sc.median_centroid_err_m is None else f"{sc.median_centroid_err_m:.2f} m"
    }

configs = ["original", "confidence_only", "evidence_improvement"]

print("\n=== ABLATION STUDY RESULTS ===")
print(f"{'Village & Configuration':<35} | {'Corr Rate':<18} | {'Med IoU':<8} | {'Improve':<8} | {'Spearman':<8} | {'Centroid Err':<12}")
print("-" * 105)

for name, res in [("Nashik (Vadnerbhairav)", v1_res), ("Malatavadi", v2_res)]:
    for config in configs:
        metrics = run_simulation(res, config)
        disp_name = f"{name} ({config})"
        print(f"{disp_name:<35} | {metrics['corr_rate']:<18} | {metrics['median_iou']:<8} | {metrics['improvement']:<8} | {metrics['calibration']:<8} | {metrics['centroid_err']:<12}")
