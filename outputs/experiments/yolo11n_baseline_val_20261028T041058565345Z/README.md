# Initial YOLO checkpoint validation

Evaluated only the 810-image validation split (1,744 boxes), using CPU because MPS is unavailable in the agent session. No training or test evaluation was performed. Checkpoint SHA-256 was unchanged after evaluation.

The standard Ultralytics internal evaluation closely reproduces the original epoch-10 metrics. Confidence 0.001 retains low-score candidates for AP evaluation. The confusion matrix uses this confidence and matching IoU 0.45; it must not be interpreted as the precision/recall operating point reported by the evaluator.

The optional external faster_coco_eval package is absent. Its evaluation was skipped; no dependencies were installed. Placeholder size-specific metrics were removed from metrics.json. The script was subsequently updated to exclude these optional size metrics automatically; environment.json retains the SHA-256 of the script actually used for this run.

Files include environment and checkpoint identity, explicit validation configuration, original dataset YAML, numeric metrics, comparison against the saved epoch-10 log, confusion matrix CSV, prediction JSON, and validation figures.
