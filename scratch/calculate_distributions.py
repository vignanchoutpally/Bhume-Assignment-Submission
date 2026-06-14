import numpy as np
from diagnostic_study import run_diagnostic_for_village

print("Loading diagnostic data...")
v1_res = run_diagnostic_for_village("data/34855_vadnerbhairav_chandavad_nashik")
v2_res = run_diagnostic_for_village("data/malatavadi")

def compute_distributions(res, name):
    plot_data = res["plot_data"]
    
    # We produce distributions for all corrected plots (before applying new safeguards)
    corrected_plots = [d for d in plot_data.values() if not d["is_geom_flagged"]]
    
    confs = [d["conf_calib"] for d in corrected_plots]
    shifts = [d["shift_dist"] for d in corrected_plots]
    improvements = [d["evidence_improvement"] for d in corrected_plots]
    
    print(f"\n=== Distributions for {name} (Corrected Plots: {len(corrected_plots)}) ===")
    
    # 1. Confidence bins
    print("\nCalibrated Confidence Distribution:")
    conf_bins = np.linspace(0.0, 1.0, 6) # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, bin_edges = np.histogram(confs, bins=conf_bins)
    for i in range(len(hist)):
        print(f"  [{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}]: {hist[i]} plots ({hist[i]/len(corrected_plots)*100:.2f}%)")
        
    # 2. Shift distance bins
    print("\nShift Distance Distribution (meters):")
    shift_bins = [0, 5, 10, 15, 20, 30]
    hist, bin_edges = np.histogram(shifts, bins=shift_bins)
    for i in range(len(hist)):
        print(f"  [{bin_edges[i]:02d} - {bin_edges[i+1]:02d}m]: {hist[i]} plots ({hist[i]/len(corrected_plots)*100:.2f}%)")
        
    # 3. Alignment improvement bins (evidence improvement)
    print("\nEvidence Improvement Distribution:")
    imp_bins = [-0.05, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20]
    hist, bin_edges = np.histogram(improvements, bins=imp_bins)
    for i in range(len(hist)):
        print(f"  [{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}]: {hist[i]} plots ({hist[i]/len(corrected_plots)*100:.2f}%)")

compute_distributions(v1_res, "Vadnerbhairav (Nashik)")
compute_distributions(v2_res, "Malatavadi")
