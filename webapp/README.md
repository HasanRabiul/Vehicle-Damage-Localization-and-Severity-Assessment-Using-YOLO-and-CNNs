# DamageLab local dashboard

From the project root:

```sh
python3 webapp/app.py
```

Open http://127.0.0.1:8120 in your browser. Stop with Ctrl+C.

The application requires Flask, Pillow, PyTorch and Ultralytics in the Python environment. It uses the existing project environment; it does not install or upgrade anything. Run only on your local Mac: the server binds to loopback and does not provide authentication for network deployment.

Features:
- Live reading of the initial YOLO training CSV, configuration, dataset counts, and saved figures.
- Pipeline A–Z: an interactive process map (steps A–N) showing how the training, validation and test splits flow through detection and severity, from raw data to the final result. Every number is read from saved outputs by `webapp/pipeline.py`; a step whose output is missing shows "not run yet". Click a step for its details, script and output folder. Tabs can be linked directly, e.g. http://127.0.0.1:8120/#process.
- Figures & curves grouped by experiment in pipeline order, with captions; only the latest run of each experiment is shown, earlier runs are folded underneath. Groups, descriptions and captions live in `webapp/catalog.py`; new figures not listed there appear under "Other saved figures".
- Searchable, paginated galleries for saved images, each with a title and description; severity crops of the test split are not listed.
- In-memory uploaded-image inference using the existing `best.pt`, MPS when available, otherwise CPU. No prediction files are saved. Downloading a rendered image is optional.
- Severity prediction (minor / moderate / severe) for uploaded images from `models/resnet50/severity_resnet50_whole_seed813.pt`, applied to the whole image; the whole-image input outperformed YOLO-crop input on validation and test. If the checkpoint is missing, severity is omitted rather than invented.
- Whole-image vs damage-crop severity results read from the latest saved `outputs/experiments/severity_eval_*` summaries (test preferred, otherwise validation).

The app has no training or dataset-editing endpoints. Original data, splits, checkpoints and artifacts are not rewritten. The checkpoint is loaded lazily on the first model test. Refresh the page to pick up changes to existing outputs. Dashboard headline metrics currently reference `yolo11n_initial`; figures and galleries discover files recursively in `runs/` and `outputs/`.

The test split is counted but its raw images are not exposed by the app. Uploaded images can still be test images: use independent images or validation images during development. Artifact provenance reflects directory names; existing prediction folders lack saved threshold metadata.
