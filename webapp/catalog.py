"""Curated grouping and captions for saved figures and image galleries. Paths are globs relative to the project root."""
import json
import re
from pathlib import Path

# Figure groups in pipeline order. The latest matching folder is shown; older matches are listed as superseded.
# all_runs groups show every matching folder (e.g. one training curve per seed).
FIGURE_GROUPS = [
    dict(glob='outputs/figures', stage='Dataset', title='CarDD annotation examples',
         text='Sample CarDD images with their damage boxes, used to check the COCO-to-YOLO conversion.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11n_initial', stage='Detection · round 1', title='YOLO11n initial training (10 epochs)',
         text='Ultralytics plots saved at the end of training on CarDD (single "damage" class, imgsz 640, batch 8). Curves are on the validation split. '
              'The confusion matrices use the plotting threshold of 0.001, which explains their large background counts (see the threshold audit).'),
    dict(glob='outputs/experiments/yolo11n_baseline_val_*/validation', stage='Detection · round 1', title='Baseline re-validation',
         text='The saved checkpoint re-validated on 810 CarDD validation images without retraining. Reproduced mAP50 0.502 and mAP50–95 0.313.'),
    dict(glob='outputs/experiments/yolo11n_threshold_audit_*', stage='Detection · round 1', title='Confidence-threshold audit',
         text='Precision, recall and false positives at fixed confidence thresholds (matching IoU > 0.45). At 0.25: precision 66%, recall 44%, 391 false positives.'),
    dict(glob='outputs/experiments/yolo11n_error_analysis_*', stage='Detection · round 1', title='Detection error analysis',
         text='Sampled validation images at confidence > 0.25. Ground truth on the left, predictions on the right. '
              'Blue: matched annotation · magenta: missed · green: matched prediction · orange: unmatched prediction.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11n_e100', stage='Detection · stronger detector', title='YOLO11n, up to 100 epochs',
         text='The same YOLO11n retrained for up to 100 epochs with early stopping (patience 20). Curves are on the validation split.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11s_e100', stage='Detection · stronger detector', title='YOLO11s, up to 100 epochs (chosen detector)',
         text='The larger YOLO11s with the same settings. Chosen on validation at confidence 0.40 and frozen for the severity pipeline.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11s_e100_repeat', stage='Detection · stronger detector', title='YOLO11s identical repeat',
         text='A second YOLO11s run with identical settings (it was meant to be the six-class run, but read the one-class labels). '
              'Kept as a measure of run-to-run variation on MPS: 0.03 validation mAP50 apart from the chosen run.'),
    dict(glob='outputs/experiments/detector_selection_*', stage='Detection · selection', title='Detector and threshold selection (validation)',
         text='Precision, recall and F1 of each candidate across confidence thresholds on the 810 validation images (IoU ≥ 0.5, one-to-one matching). '
              'Each model keeps its best-F1 threshold; YOLO11s at 0.40 is chosen.'),
    dict(glob='outputs/experiments/detector_test_*/*', stage='Detection · test (once)', title='Detector test evaluation', all_runs=True, label='folder',
         text='Ultralytics curves of the chosen detector and the round-1 YOLO11n on the 374 CarDD test images, evaluated once after selection.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11s_6cls_e100', stage='Detection · six-class study', title='YOLO11s with six damage types',
         text='Training plots of the six-class model. Its "all" mAP is the mean over the six types and is not comparable with the one-class models; '
              'see the class-agnostic comparison in the Findings tab.'),
    dict(glob='outputs/experiments/six_class_val_*/*', stage='Detection · six-class study', title='Six-class vs one-class · validation', all_runs=True, label='folder',
         text='Class-agnostic validation runs (types merged) for the one-class runs and the six-class model, plus the six-class per-type evaluation with its type confusion matrix.'),
    dict(glob='outputs/experiments/six_class_test_*/*', stage='Detection · six-class study', title='Six-class vs one-class · test (once)', all_runs=True, label='folder',
         text='The same comparison run once on CarDD test with the validation thresholds. Nothing was chosen from it.'),
    dict(glob='outputs/experiments/severity_data3a_audit_*', stage='Severity · data', title='Severity dataset audit (data3a)',
         text='Duplicate, leakage and CarDD-overlap checks. Cross-split copies and label conflicts are excluded; validation/test images found in CarDD are excluded.'),
    dict(glob='outputs/experiments/severity_crops_*', stage='Severity · inputs', title='YOLO damage crops for severity',
         text='Final localized input: union of YOLO11s detections (confidence ≥ 0.40) padded 30% per side; whole image when nothing is detected. '
              'Earlier variants are folded below: round 1 (YOLO11n ≥ 0.25, 10%), YOLO11s ≥ 0.40 with 10%, YOLO11s ≥ 0.25 with 10%.'),
    dict(glob='outputs/experiments/severity_resnet50_*', stage='Severity · training', title='ResNet50 training curves', all_runs=True,
         text='Loss, accuracy and validation macro-F1 per epoch for each input, crop variant and seed (date in the caption). Dotted line: backbone unfrozen. Dashed line: selected epoch.'),
    dict(glob='outputs/experiments/severity_eval_validation_*', stage='Severity · validation', title='Severity evaluation · validation',
         text='Final frozen configuration on 239 validation images (round-1 versions folded below). Optimistic: epochs and the crop variant were selected on validation.'),
    dict(glob='outputs/experiments/severity_eval_test_*', stage='Severity · test (once)', title='Severity evaluation · test (final)',
         text='The frozen configuration on 245 test images: whole 0.583 vs localized 0.568 macro-F1, ΔF1 −0.015 [−0.061, +0.030]. '
              'The round-1 test (whole 0.583 vs YOLO11n crops 0.514) is folded below.'),
    dict(glob='outputs/experiments/severity_error_analysis_*', stage='Severity · error analysis', title='Grad-CAM and moderate-class errors',
         text='Analysis of the fixed test predictions; nothing was chosen. Grad-CAM of the whole-image model with the YOLO11s boxes (attention ratio 1.2: '
              'it already looks at damage, including damage YOLO missed), and moderate images the whole-image model got wrong.'),
    dict(glob='runs/detect/val', stage='Detection · earlier', title='Earlier Ultralytics validation run',
         text='A validation run saved before the controlled experiments. Its settings were not recorded, so the baseline re-validation replaces it.'),
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
    'threshold_selection.png': 'Precision, recall and F1 by threshold, per candidate',
    'gradcam.png': 'Grad-CAM of the whole-image model with YOLO boxes',
    'moderate_errors.png': 'Misclassified moderate images with YOLO boxes',
}

# Image galleries. Severity test crops are deliberately not listed.
GALLERY_GROUPS = [
    dict(glob='outputs/experiments/yolo11n_baseline_val_*/validation', stage='Detection', title='Step 1 re-validation batches',
         text='Ultralytics validation mosaics from the step 1 re-validation: ground-truth labels and predictions for the first three batches.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11n_initial', stage='Detection', title='Training mosaics and validation batches',
         text='Augmented training batches and validation labels vs predictions saved during the initial 10-epoch training.'),
    dict(glob='runs/detect/runs/damage_detection/yolo11*_e100*', stage='Detection · stronger detector', title='{label} · training mosaics and validation batches',
         text='Augmented training batches and validation labels vs predictions saved during training of {label}.'),
    dict(glob='outputs/experiments/six_class_val_*/*_per_type', stage='Detection · six-class study', title='Six-class predictions on validation',
         text='Validation labels vs predictions of the six-class model, with the damage type on each box.'),
    dict(glob='runs/detect/outputs/damage_predictions', stage='Detection', title='Predictions on CarDD validation (run 1)',
         text='Rendered predictions on all 810 validation images, saved before the controlled experiments. The confidence threshold was not recorded.'),
    dict(glob='runs/detect/outputs/damage_predictions-2', stage='Detection', title='Predictions on CarDD validation (run 2)',
         text='A second rendering of the same 810 validation images. The confidence threshold was not recorded.'),
    dict(glob='runs/detect/outputs/damage_predictions/crops/damage', stage='Detection', title='Detected damage crops (run 1)',
         text='Every detected box from prediction run 1 cut out as its own image.'),
    dict(glob='outputs/experiments/severity_crops_*/crops/validation/*', stage='Severity crops · validation', png=True,
         title='{label} · validation · {variant}', text='Crops the severity classifier sees for validation images labelled {label} ({variant}). Fallback crops are the whole image.'),
    dict(glob='outputs/experiments/severity_crops_*/crops/training/*', stage='Severity crops · training', png=True,
         title='{label} · training · {variant}', text='Crops used to train the localized classifier for images labelled {label} ({variant}). Fallback crops are the whole image.'),
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
                    if g.get('label') == 'folder':
                        caption = f"{d.name.replace('_', ' ')} · {caption}"
                    elif g.get('all_runs'):
                        parts = d.name.split('_')  # severity_resnet50_<variant>_seed<n>_<stamp>
                        caption = f'{parts[2]} input · seed {parts[3][4:]} · {parts[4][4:6]}/{parts[4][6:8]}'
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


def _test_images(root, d):
    """Folders whose images come from a test split (raw test images, or Ultralytics batches rendered on test)."""
    parts = d.relative_to(root).parts
    return 'test' in parts or any(p.startswith(('detector_test_', 'six_class_test_')) for p in parts)


def _crop_variant(root, d):
    """Detector, threshold and padding of the severity_crops_* folder containing d, e.g. 'yolo11s_e100 ≥ 0.4 · pad 30%'."""
    exp = next((p for p in [d, *d.parents] if p.name.startswith('severity_crops_')), None)
    cfg = exp and exp / 'config.json'
    if not cfg or not cfg.exists():
        return ''
    c = json.loads(cfg.read_text())
    pad = re.search(r'padded (\d+)%', c.get('crop', ''))
    return f"{c['checkpoint'].split('/')[-3]} ≥ {c['conf']} · pad {pad.group(1) if pad else '?'}%"


def gallery_groups(root):
    seen, groups = set(), []
    for g in GALLERY_GROUPS:
        ext = IMAGE + (('.png',) if g.get('png') else ())
        for d in sorted(p for p in root.glob(g['glob']) if p.is_dir()):
            if d in seen or _test_images(root, d):
                continue
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in ext)
            if not n:
                continue
            seen.add(d)
            fmt = dict(label=_pretty(d.name), variant=_crop_variant(root, d))
            groups.append(dict(folder=_rel(root, d), stage=g['stage'], title=g['title'].format(**fmt),
                               text=g['text'].format(**fmt), count=n, png=bool(g.get('png'))))
    for d in sorted({p.parent for base in ['runs', 'outputs'] for p in (root / base).rglob('*') if p.suffix.lower() in IMAGE}):
        if d not in seen and not _test_images(root, d):
            groups.append(dict(folder=_rel(root, d), stage='Other', title=_rel(root, d), text='Not yet described.',
                               count=sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE), png=False))
    return groups


def gallery_extensions(root, folder):
    g = next((x for x in gallery_groups(root) if x['folder'] == folder), None)
    return None if g is None else IMAGE + (('.png',) if g['png'] else ())
