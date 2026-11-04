# Validation confidence-threshold audit

Installed Ultralytics ConfusionMatrix; one-to-one confidence-ordered matching; strict score > threshold and IoU > threshold.

Saved JSON rounds boxes to 3 decimals and confidence to 5 decimals. Thresholds operate on frozen post-NMS predictions, not fresh inference. No AP recalculation or threshold selection.

All six CarDD damage types are treated as damage. No severity labels are inferred. No training or test-set evaluation occurred.

At confidence 0.001 and matching IoU 0.45: TP=1676, FP=120956, FN=68.
Source in-memory confusion matrix: [[1675.0, 121249.0], [69.0, 0.0]]. Differences arise from rounded JSON scores/coordinates and reconstruction of label coordinates. Scores rounded to exactly 0.001 are excluded by the strict filter; this export cannot exactly reconstruct the in-memory matrix.

The low-confidence matrix and the headline validation precision/recall measure different operating points. AP uses ranked predictions across confidence thresholds; these fixed-threshold precision/recall values are not AP.
Background in a detection confusion matrix represents unmatched boxes or annotations. The bottom-right cell is not a true-negative count.
The validation split contains no fully unannotated background images; this analysis does not establish performance on an undamaged-vehicle dataset.

No operational threshold is selected here. Choose a crop-generation policy using validation data only after reviewing errors and no-detection cases.
