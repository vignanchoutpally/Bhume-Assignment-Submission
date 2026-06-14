# ChatGPT Transcript (https://chatgpt.com/c/6a2c2c93-fbc8-83ee-835b-d7cc60212203) – Boundary Correction Challenge

## Session 1 – Understanding the Challenge

### Objective

Analyze historical cadastral parcel boundaries and automatically correct georeferencing errors using imagery and boundary evidence.

### Initial Understanding

The challenge provides:

* input.geojson
* imagery.tif
* boundaries.tif
* example_truths.geojson
* starter kit

Expected output:

* predictions.geojson

Each parcel must be:

* CORRECTED
* FLAGGED

with an associated confidence score.

### Key Observation

The challenge emphasizes:

1. Accuracy
2. Confidence calibration
3. Restraint
4. Generalization

Confidence calibration was identified as the most important evaluation component.

---

## Session 2 – Dataset Analysis

### Vadnerbhairav Analysis

Initial inspection showed:

* 2457 plots
* 6 public truth plots
* imagery raster
* boundary raster

Comparison against example truths suggested:

* Most errors were positional rather than geometric.
* Shapes were often reasonable.
* Typical centroid displacement ranged between approximately 10–20 meters.

### Hypothesis

Most fixable parcels suffer from:

* georeferencing drift
* translation errors
* small rotational errors

rather than completely incorrect parcel shapes.

### Recommended Direction

Treat the task as:

Boundary Alignment

instead of:

Boundary Reconstruction

---

## Session 3 – First Pipeline Design

### Proposed Workflow

Input Parcel
↓
Generate Candidate Shifts
↓
Score Boundary Alignment
↓
Estimate Confidence
↓
CORRECT or FLAG

### Candidate Search

Search nearby translations and small rotations around each parcel.

### Scoring Signals

* boundary raster alignment
* image edge alignment
* parcel area consistency
* neighboring parcel consistency

### Confidence Signals

* improvement over original
* best-vs-second-best candidate margin
* area consistency
* movement distance

---

## Session 4 – Evaluation Results

### First Public Evaluation

Results:

* Median IoU = 0.750
* Improvement = +0.131
* Accurate@0.5 = 100%
* Calibration = 0.10

### Interpretation

Geometry correction worked well.

Main weakness:

Confidence calibration.

### Recommendation

Focus future improvements on confidence estimation rather than geometry optimization.

---

## Session 5 – Confidence Calibration Discussion

### Identified Problem

Confidence values were not sufficiently correlated with actual correction quality.

### Recommended Improvements

Use multiple confidence signals:

confidence =
gain

* candidate margin
* boundary evidence
* area consistency
* topology consistency

### Additional Recommendation

Increase use of FLAGGED status for uncertain cases.

---

## Session 6 – Generalization Failure Discovery

### New Village: Malatavadi

Public evaluation showed:

* Median IoU = 0.145
* Improvement = -0.106

while Vadnerbhairav remained strong.

### Interpretation

The method did not generalize.

Likely cause:

The optimizer was forcing corrections in a village where evidence was weak or inconsistent.

### Recommendation

Introduce a:

Move vs Do-Not-Move

decision stage.

---

## Session 7 – Do-No-Harm Principle

### Proposed Strategy

A correction should only be accepted when:

Corrected Confidence

>

Original Confidence

and supporting evidence improves.

### Suggested Safeguards

* evidence improvement threshold
* confidence threshold
* large-shift penalty
* correction rejection logic

### Goal

Avoid making parcels worse.

---

## Session 8 – Evidence Improvement Safeguard

### Diagnostic Findings

Vadnerbhairav:

* strong evidence gains
* consistent offsets

Malatavadi:

* weak evidence gains
* non-uniform offsets

### Recommendation

Accept correction only if:

ΔE ≥ threshold

where:

ΔE = corrected evidence − original evidence

Additional safeguards:

* confidence threshold
* large-shift safeguard

---

## Session 9 – Updated Results

### Vadnerbhairav

* 3 corrected
* 3 flagged
* Median IoU = 0.872
* Improvement = +0.194
* Calibration = 1.00

### Malatavadi

* 1 corrected
* 2 flagged
* Median IoU = 0.574
* Improvement positive
* Accurate@0.5 = 100%

### Interpretation

The system now:

* corrects when evidence is strong
* flags when evidence is weak

This behavior aligns with the challenge goals of:

* calibration
* restraint
* generalization

---

## Final Submission Recommendation

Recommended submission:

Evidence-Improvement Safeguarded Pipeline

Key characteristics:

1. Candidate alignment search
2. Boundary and edge evidence scoring
3. Confidence calibration
4. Evidence-improvement safeguard
5. Large-shift safeguard
6. Conservative correction acceptance

Final philosophy:

Correct only when evidence strongly supports improvement; otherwise preserve the official boundary and flag the parcel.
