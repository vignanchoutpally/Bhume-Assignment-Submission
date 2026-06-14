# BhuMe Boundary Correction Challenge Solution

A confidence-aware geospatial boundary correction system for cadastral plot alignment using satellite imagery, boundary hints, spatial consistency constraints, and calibrated uncertainty estimation.

The solution is designed to improve misaligned cadastral boundaries while explicitly identifying plots where a reliable correction cannot be made. The final pipeline prioritizes correction quality, confidence calibration, and restraint over aggressive correction rates.

---

# Repository Structure

```text
repo/
│
├── src/
│   ├── __init__.py
│   ├── io.py
│   ├── alignment.py
│   ├── scoring.py
│   ├── confidence.py
│   ├── evaluate.py
│   ├── predict.py
│   └── baseline.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── data/
│   ├── 34855_vadnerbhairav_chandavad_nashik/
│   │   ├── input.geojson
│   │   ├── imagery.tif
│   │   ├── boundaries.tif
│   │   └── predictions.geojson
│   │
│   └── malatavadi/
│       ├── input.geojson
│       ├── imagery.tif
│       ├── boundaries.tif
│       └── predictions.geojson
│
├── experiments/
│   ├── scorecard_nashik.txt
│   ├── scorecard_malatavadi.txt
│   ├── ranked_corrections_nashik.txt
│   ├── ranked_corrections_malatavadi.txt
│   └── diagnostic reports
│
├── transcripts/
│   ├── README.md
│   ├── antigravity_transcript.md
|   └── chatgpt_transcript.md
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# Problem Statement

The task is to automatically correct cadastral plot boundaries using:

* Official cadastral polygons (`input.geojson`)
* Satellite imagery (`imagery.tif`)
* Optional boundary hints (`boundaries.tif`)

For each plot, the system must either:

* Produce a corrected boundary (`CORRECTED`)
* Indicate insufficient confidence (`FLAGGED`)

along with a calibrated confidence score.

The challenge explicitly rewards:

* Accurate corrections
* Honest confidence estimates
* Restraint when evidence is weak
* Generalization across villages

---

# Methodology

## 1. Automated Global Georeferencing Estimation

For unseen villages, no ground-truth boundaries are available.

To estimate village-wide georeferencing drift:

1. Select the largest plots in the village.
2. Sample boundary coordinates.
3. Search a coarse translation grid (±30m).
4. Evaluate overlap with boundary evidence.
5. Select the shift with the strongest aggregate evidence.

This provides an initial alignment estimate without requiring example truths.

---

## 2. Continuous Potential Fields

Direct comparison against binary boundary rasters is highly sensitive to small pixel errors.

To make optimization smoother:

### Boundary Layer

* Convert boundary hints into a continuous potential field.
* Apply Gaussian blur (σ = 2.0).

### Imagery Layer

* Convert imagery to grayscale.
* Compute Sobel gradients.
* Apply Gaussian smoothing (σ = 1.0).
* Normalize gradient magnitudes.

This transforms discrete edges into continuous optimization surfaces that are more robust to noise and small positional errors.

---

## 3. Vectorized Candidate Search

Each plot is represented using sampled boundary coordinates.

Candidate transformations are generated across:

* Translation X
* Translation Y
* Rotation

Instead of repeatedly performing polygon operations:

* Coordinates are transformed using NumPy vectorized operations.
* Pixel-space lookups are performed directly against precomputed raster layers.
* Candidate evaluation remains computationally efficient despite large search spaces.

A hierarchical search strategy is used:

1. Coarse translation search
2. Local refinement
3. Fine rotation optimization

---

## 4. Spatial Consistency Regularization

Adjacent plots should generally move in similar directions.

To preserve local topology:

1. Spatial neighbors are identified.
2. Neighbor displacement statistics are computed.
3. Candidate scores are penalized when they differ substantially from surrounding plots.

This reduces:

* Topology violations
* Plot overlaps
* Isolated incorrect corrections

---

# Confidence Calibration

A confidence score in the range [0,1] is computed using multiple independent signals.

## Alignment Strength

Measures how strongly the corrected boundary aligns with:

* Boundary hints
* Imagery gradients

---

## Improvement Margin

Measures how much better the selected correction performs relative to the original plot location.

Weak improvements receive lower confidence.

---

## Peak Uniqueness

Multiple candidate alignments may produce similar scores.

The difference between:

* Best candidate
* Second-best candidate

is used as an ambiguity estimate.

Small margins indicate uncertainty and reduce confidence.

---

## Neighbor Consistency

Corrections that disagree strongly with nearby plots receive lower confidence.

---

## Area Consistency

The corrected geometry is compared against official area records.

Large discrepancies indicate potential shape or digitization issues and reduce confidence.

---

## Overconfidence Prevention

A confidence of exactly 1.0 is rarely justified in geospatial alignment tasks.

To encourage honest uncertainty:

* High confidence values are compressed.
* Maximum confidence is capped below 1.0.

This improves calibration and avoids overconfident predictions.

---

# Do-No-Harm Safeguards

During development, two distinct village behaviors were observed:

### Vadnerbhairav (Nashik)

The village exhibited a largely uniform georeferencing offset.

Global alignment performed well and produced strong corrections.

### Malatavadi

The village exhibited non-uniform local offsets.

Aggressive correction sometimes degraded already-reasonable boundaries.

To address this, the final pipeline introduces evidence-based correction acceptance.

A correction is accepted only when:

* Confidence is sufficiently high.
* Evidence improves relative to the original geometry.
* Spatial consistency is maintained.

Otherwise:

* The plot is flagged.
* The original geometry is preserved.

This ensures the system avoids harmful corrections when evidence is insufficient.

---

# Overlap Validation

After all corrections are generated:

1. Neighbor intersections are evaluated.
2. Excessive overlap is detected.
3. Conflicting corrections are reverted.

Reverted plots:

* Become FLAGGED
* Retain original geometry
* Receive confidence 0.0

This preserves cadastral topology.

---

# Generalization Strategy

The final system is designed to operate on unseen villages without village-specific tuning.

Key design principles:

* Automatic global alignment estimation
* Dynamic coordinate system selection
* Confidence-aware correction acceptance
* Evidence-based safeguards
* No manual geometry editing
* No village-specific thresholds

The same pipeline was used across all villages.

---

# Public-Test Results

## Vadnerbhairav (Nashik)

| Metric                    | Result |
| ------------------------- | ------ |
| Corrected                 | 3      |
| Flagged                   | 3      |
| Median IoU                | 0.872  |
| Improvement over official | +0.194 |
| Accurate @ IoU ≥ 0.5      | 100%   |
| Median centroid error     | 5.2 m  |
| Calibration (Spearman ρ)  | 1.00   |

---

## Malatavadi

| Metric                | Result |
| --------------------- | ------ |
| Corrected             | 1      |
| Flagged               | 2      |
| Median IoU            | 0.574  |
| Official baseline IoU | 0.510  |
| Accurate @ IoU ≥ 0.5  | 100%   |
| Median centroid error | 4.7 m  |

These results demonstrate both:

* Effective correction when strong evidence exists.
* Appropriate restraint when evidence is insufficient.

---

# Lessons Learned

The most important lesson from this challenge was that accuracy alone is insufficient.

A method must also know when not to act.

Early versions of the pipeline performed well on villages with near-uniform offsets but degraded performance on villages exhibiting non-uniform local distortions.

Through diagnostic studies, threshold evaluation, confidence calibration, and safeguard design, the final system evolved from an aggressive optimizer into a confidence-aware correction framework that prioritizes reliable improvements over correction volume.

This aligns closely with the challenge objective:

> Correct the plots. Know when you can't.

---

# AI-Assisted Development

AI tools were used extensively throughout the project for:

* understanding the geospatial alignment problem,
* brainstorming correction strategies,
* confidence calibration design,
* diagnostic analysis,
* safeguard development,
* implementation support,
* code refactoring and validation.

Development transcripts and conversation logs are included in the `transcripts/` directory as requested.

---

# Setup

Install dependencies using:

```bash
uv sync
```

---

# Running the Pipeline

## Vadnerbhairav

```bash
uv run python src/predict.py data/34855_vadnerbhairav_chandavad_nashik
```

Output:

```text
data/34855_vadnerbhairav_chandavad_nashik/predictions.geojson
```

---

## Malatavadi

```bash
uv run python src/predict.py data/malatavadi
```

Output:

```text
data/malatavadi/predictions.geojson
```

---

# Diagnostics

When executed, the pipeline automatically generates diagnostics including:

* correction counts
* flagging statistics
* confidence distributions
* overlap rejections
* ranked corrections
* calibration diagnostics

These artifacts were used during development and are included in the repository for transparency and reproducibility.
