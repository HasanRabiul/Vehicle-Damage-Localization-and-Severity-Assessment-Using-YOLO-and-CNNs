"""Curated grouping and captions for saved figures and image galleries. Paths are globs relative to the project root."""
from pathlib import Path

# Figure groups in pipeline order. The latest matching folder is shown; older matches are listed as superseded.
# all_runs groups show every matching folder (e.g. one training curve per seed).
FIGURE_GROUPS = [
    dict(glob='outputs/figures', stage='Dataset', title='CarDD annotation examples',
         text='Sample CarDD images with their damage boxes, used to check the COCO-to-YOLO conversion.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11n_initial', stage='Detection · training', title='YOLO11n initial training (10 epochs)',
         text='Ultralytics plots saved at the end of training on CarDD (single "damage" class, imgsz 640, batch 8). Curves are on the validation split. '
              'The confusion matrices use the plotting threshold of 0.001, which explains their large background counts (see step 2).'),
    dict(glob='outputs/experiments/yolo11n_baseline_val_*/validation', stage='Detection · step 1', title='Baseline re-validation',
         text='The saved checkpoint re-validated on 810 CarDD validation images without retraining. Reproduced mAP50 0.502 and mAP50–95 0.313.'),
    dict(glob='outputs/experiments/yolo11n_threshold_audit_*', stage='Detection · step 2', title='Confidence-threshold audit',
         text='Precision, recall and false positives at fixed confidence thresholds (matching IoU > 0.45). At 0.25: precision 66%, recall 44%, 391 false positives.'),
    dict(glob='outputs/experiments/yolo11n_error_analysis_*', stage='Detection · step 3', title='Detection error analysis',
         text='Sampled validation images at confidence > 0.25. Ground truth on the left, predictions on the right. '
              'Blue: matched annotation · magenta: missed · green: matched prediction · orange: unmatched prediction.'),
    dict(glob='outputs/experiments/severity_data3a_audit_*', stage='Severity · step 4', title='Severity dataset audit (data3a)',
         text='Duplicate, leakage and CarDD-overlap checks. Cross-split copies and label conflicts are excluded; validation/test images found in CarDD are excluded.'),
    dict(glob='outputs/experiments/severity_crops_*', stage='Severity · step 5', title='YOLO damage crops for severity',
         text='Union of detections (confidence > 0.25) padded 10% per side; whole image when nothing is detected (about half of the images).'),
    dict(glob='outputs/experiments/severity_resnet50_*', stage='Severity · step 5', title='ResNet50 training curves', all_runs=True,
         text='Loss, accuracy and validation macro-F1 per epoch for each input variant and seed. Dotted line: backbone unfrozen. Dashed line: selected epoch.'),
    dict(glob='outputs/experiments/severity_eval_validation_*', stage='Severity · step 5', title='Severity evaluation · validation',
         text='All six classifiers on 239 validation images. Whole → whole macro-F1 0.683 vs crop → crop 0.646 (mean of 3 seeds). Optimistic: epochs were selected on validation.'),
    dict(glob='outputs/experiments/severity_eval_test_*', stage='Severity · step 5', title='Severity evaluation · test (final)',
         text='Single pre-approved evaluation on 245 test images. Whole → whole macro-F1 0.583 vs crop → crop 0.514; paired difference 95% CI [+0.016, +0.121].'),
    dict(glob='runs/detect/val', stage='Detection · earlier', title='Earlier Ultralytics validation run',
         text='A validation run saved before the controlled experiments. Its settings were not recorded, so step 1 replaces it.'),
]

FIGURE_CAPTIONS = {
    'cardd_annotations.png': 'Annotated CarDD samples',
    'results.png': 'Losses and validation metrics by epoch',
    'BoxPR_curve.png': 'Precision–recall curve',
    'BoxF1_curve.png': 'F1 vs confidence',
    'BoxP_curve.png': 'Precision vs confidence',
    'BoxR_curve.png': 'Recall vs confidence',
    'confusion_matrix.png': 'Confusion matrix (counts)',
    'confusion_matrix_normalized.png': 'Confusion matrix (normalised)',
    'confidence_analysis.png': 'Precision, recall and false positives by threshold',
    'confusion_threshold_comparison.png': 'Confusion matrices at selected thresholds',
    'correct_localization.png': 'Correct localization',
    'false_positives.png': 'False positives',
    'missed_damage.png': 'Missed damage',
    'no_detections.png': 'No detections',
    'correct_localization_contact_sheet.png': 'Correct localization (first version)',
    'false_positives_contact_sheet.png': 'False positives (first version)',
    'missed_damage_contact_sheet.png': 'Missed damage (first version)',
    'no_detections_contact_sheet.png': 'No detections (first version)',
    'cross_split_duplicates.png': 'Duplicates across splits',
    'label_conflicts.png': 'Same image, conflicting labels',
    'cardd_near_matches.png': 'data3a images also in CarDD',
    'crop_samples.png': 'Detections, crops and fallbacks per class',
    'curves.png': 'Training curves',
    'cross_input_macro_f1.png': 'Macro-F1: training input × evaluation input',
    'confusion_matrices.png': 'Confusion matrices (3 seeds summed)',
    'errors_whole.png': 'Whole-image model: correct and wrong examples',
    'errors_crop.png': 'Crop model: correct and wrong examples',
}

# Image galleries. Severity test crops are deliberately not listed.
GALLERY_GROUPS = [
    dict(glob='outputs/experiments/yolo11n_baseline_val_*/validation', stage='Detection', title='Step 1 re-validation batches',
         text='Ultralytics validation mosaics from the step 1 re-validation: ground-truth labels and predictions for the first three batches.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11n_initial', stage='Detection', title='Training mosaics and validation batches',
         text='Augmented training batches and validation labels vs predictions saved during the initial 10-epoch training.'),
    dict(glob='runs/detect/outputs/damage_predictions', stage='Detection', title='Predictions on CarDD validation (run 1)',
         text='Rendered predictions on all 810 validation images, saved before the controlled experiments. The confidence threshold was not recorded.'),
    dict(glob='runs/detect/outputs/damage_predictions-2', stage='Detection', title='Predictions on CarDD validation (run 2)',
         text='A second rendering of the same 810 validation images. The confidence threshold was not recorded.'),
    dict(glob='runs/detect/outputs/damage_predictions/crops/damage', stage='Detection', title='Detected damage crops (run 1)',
         text='Every detected box from prediction run 1 cut out as its own image.'),
    dict(glob='outputs/experiments/severity_crops_*/crops/validation/*', stage='Severity crops · validation', png=True,
         title='{label} · validation', text='YOLO crops the severity classifier sees for validation images labelled {label}. Fallback crops are the whole image.'),
    dict(glob='outputs/experiments/severity_crops_*/crops/training/*', stage='Severity crops · training', png=True,
         title='{label} · training', text='YOLO crops used to train the crop-input classifier for images labelled {label}. Fallback crops are the whole image.'),
    dict(glob='runs/detect/val', stage='Detection · earlier', title='Earlier validation batches',
         text='Validation mosaics from an earlier Ultralytics run whose settings were not recorded.'),
    dict(glob='outputs/experiments/yolo11n_error_analysis_*/*', stage='Detection · earlier', title='Step 3 first version · {label}',
         text='Individual images from the first version of the step 3 error analysis, replaced by the grouped panels under Figures & curves.'),
    dict(glob='runs/detect/predict*', stage='Setup check', title='Sample prediction ({label})',
         text='Ultralytics sample image used to check the installation. Not part of the experiments.'),
]

IMAGE = ('.jpg', '.jpeg')


def _rel(root, p):
    return str(p.relative_to(root))


def _pretty(name):
    return name.split('-', 1)[-1].replace('_', ' ') if name[:2].isdigit() else name.replace('_', ' ')


def figure_groups(root):
    claimed, groups = set(), []
    for g in FIGURE_GROUPS:
        folders = sorted(p for p in root.glob(g['glob']) if p.is_dir())
        current = folders if g.get('all_runs') else folders[-1:]
        figs = lambda d: sorted(d.glob('*.png'))
        def items(dirs):
            out = []
            for d in dirs:
                for f in figs(d):
                    claimed.add(f)
                    caption = FIGURE_CAPTIONS.get(f.name, f.stem.replace('_', ' '))
                    if g.get('all_runs'):
                        parts = d.name.split('_')  # severity_resnet50_<variant>_seed<n>_<stamp>
                        caption = f'{parts[2]} input · seed {parts[3][4:]}'
                    out.append(dict(path=_rel(root, f), caption=caption))
            return out
        figures = items(current)
        if not figures:
            continue
        superseded = items(folders[:-1]) if not g.get('all_runs') else []
        groups.append(dict(stage=g['stage'], title=g['title'], text=g['text'], figures=figures, superseded=superseded,
                           source=', '.join(_rel(root, d) for d in current) if len(current) == 1 else f'{len(current)} runs',
                           superseded_source=[_rel(root, d) for d in folders[:-1]] if superseded else []))
    other = [p for base in ['runs', 'outputs'] for p in (root / base).rglob('*.png')
             if p not in claimed and 'crops' not in p.relative_to(root).parts and '.matplotlib' not in p.parts]
    if other:
        groups.append(dict(stage='Other', title='Other saved figures', text='Figures not yet assigned to an experiment group.',
                           figures=[dict(path=_rel(root, p), caption=p.name) for p in sorted(other)], superseded=[], source='', superseded_source=[]))
    return groups


def gallery_groups(root):
    seen, groups = set(), []
    for g in GALLERY_GROUPS:
        ext = IMAGE + (('.png',) if g.get('png') else ())
        for d in sorted(p for p in root.glob(g['glob']) if p.is_dir()):
            if d in seen or 'test' in d.relative_to(root).parts:
                continue
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in ext)
            if not n:
                continue
            seen.add(d)
            label = _pretty(d.name)
            groups.append(dict(folder=_rel(root, d), stage=g['stage'], title=g['title'].format(label=label),
                               text=g['text'].format(label=label), count=n, png=bool(g.get('png'))))
    for d in sorted({p.parent for base in ['runs', 'outputs'] for p in (root / base).rglob('*') if p.suffix.lower() in IMAGE}):
        if d not in seen and 'test' not in d.relative_to(root).parts:
            groups.append(dict(folder=_rel(root, d), stage='Other', title=_rel(root, d), text='Not yet described.',
                               count=sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE), png=False))
    return groups


def gallery_extensions(root, folder):
    g = next((x for x in gallery_groups(root) if x['folder'] == folder), None)
    return None if g is None else IMAGE + (('.png',) if g['png'] else ())
