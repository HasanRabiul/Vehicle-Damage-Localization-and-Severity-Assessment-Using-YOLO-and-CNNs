# Error analysis of the final test comparison

Analysis of the fixed test predictions in outputs/experiments/severity_eval_test_20261122T203029685325Z; no model or setting was chosen here. Grad-CAM uses the whole-image models on test images where YOLO11s found damage; the ratio is the share of attention inside the boxes divided by the boxes' share of the view.

## Visual inspection (author)
- gradcam.png: Grad-CAM usually highlights visible damage, but in most of the 8 samples that damage lies outside the YOLO11s boxes; only 2 heatmaps fall mainly inside the boxes.
- moderate_errors.png: several "misclassified" moderate images look plausibly correct for the predicted class (ambiguous labels), and the YOLO boxes often miss part of the damage.
