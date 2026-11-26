# DamageLab local dashboard

From the project root:

```sh
python3 webapp/app.py
```

Open http://127.0.0.1:8120 in your browser. Stop with Ctrl+C.

The application requires Flask, Pillow, PyTorch and Ultralytics in the Python environment. It uses the existing project environment; it does not install or upgrade anything. Run only on your local Mac: the server binds to loopback and does not provide authentication for network deployment.

Features:
- **Overview:** the answer to the research question (localized − whole ΔF1 and ΔR moderate with 95% intervals, round 1 vs final), headline test metrics of the detector and the severity comparison, and validation mAP50 curves of every detector run.
- **Findings:** every result of the project in order, read from saved outputs by `webapp/findings.py`: detector selection on validation and the single CarDD test; the six-class study (class-agnostic comparison, recall and AP by damage type, run-to-run variation); the localized-input variants compared on validation; the final data3a test with bootstrap intervals; the error analysis (per-class metrics, detected/fallback and crop-size subsets, moderate-class errors, Grad-CAM); conclusions and limitations.
- **Metrics explained:** every measure used in the project, with its formula, how it relates to the others and its role in this project, filled in with the saved results (`webapp/static/measures.js`). Detection: IoU, confidence and NMS, TP/FP/FN, precision and recall, F1, AP/mAP50/mAP50–95, recall by damages per image. Severity: confusion matrix, accuracy, per-class metrics, macro-F1, ΔF1 and ΔR moderate, softmax. Statistics: seeds, paired bootstrap and 95% intervals, validation vs test, post-hoc subsets, Grad-CAM ratio, crop area. The interactive parts replay the saved CarDD validation threshold sweep and recompute the confusion matrix and bootstrap from the saved final test predictions (labels only, no images).
- **Pipeline A–Z:** an interactive process map (steps A–P) showing how the training, validation and test splits flow through detection and severity, from raw data to the final result and the error analysis. Built by `webapp/pipeline.py`; a step whose output is missing shows "not run yet". Tabs can be linked directly, e.g. http://127.0.0.1:8120/#findings.
- **Test model:** in-memory inference on an uploaded image (MPS when available, otherwise CPU; nothing is saved). The frozen detector from `final_configs.json` (YOLO11s, confidence 0.40) builds the localized crop exactly as `src/prepare_severity_crops.py` does (union of boxes, padded 30%, whole image when nothing is detected); the whole-image and the localized ResNet50 then run side by side. For each input the seed with the best validation macro-F1 is used. Boxes can be displayed from the pipeline detector or from the six-class detector, which labels damage types; the display sliders never change the crop.
- **Figures & curves:** grouped by experiment in pipeline order, with captions; only the latest run of each experiment is shown, earlier runs are folded underneath. Groups, descriptions and captions live in `webapp/catalog.py`; new figures not listed there appear under "Other saved figures".
- **Prediction gallery:** searchable, paginated galleries for saved images, including the crops of every localized-input variant. Images from test splits (raw images, test crops, and batches rendered on test) are not listed.
- **Experiment details:** checkpoint paths, the frozen `final_configs.json`, the detector training configuration, and links to saved CSV/JSON/YAML files.

The app has no training or dataset-editing endpoints. Original data, splits, checkpoints and artifacts are not rewritten. Models are loaded lazily on the first prediction. Refresh the page to pick up changes to existing outputs.

The test split is counted but its raw images are not exposed by the app. Uploaded images can still be test images: use independent images or validation images during development. Artifact provenance reflects directory names; existing prediction folders lack saved threshold metadata.
