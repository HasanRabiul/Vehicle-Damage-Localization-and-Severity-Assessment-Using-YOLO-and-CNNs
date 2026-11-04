"""A-to-Z process map: every step of detection and severity, placed by data split, with numbers read from saved outputs.

Layout: columns are process stages, rows are split lanes (training / validation / test) for each track. A missing
output leaves its node marked 'not run yet' instead of inventing a value.
"""
import csv
import json
from collections import defaultdict

COLUMNS = ['Data', 'Prepare', 'Build inputs', 'Train', 'Validate & select', 'Test (once)', 'Result']
LANES = [('detection', 'training'), ('detection', 'validation'), ('detection', 'test'),
         ('severity', 'training'), ('severity', 'validation'), ('severity', 'test')]


def _latest(root, pattern, file):
    runs = sorted(p for p in (root / 'outputs/experiments').glob(pattern) if (p / file).exists())
    return runs[-1] if runs else None


def _json(path):
    return json.loads(path.read_text()) if path and path.exists() else None


def _rel(root, p):
    return str(p.relative_to(root)) if p else None


def _yaml_value(path, key):
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.startswith(key + ':'):
            return line.split(':', 1)[1].strip()
    return None


def pipeline(root):
    run = root / 'runs/detect/runs/damage_detection/yolo11n_initial'
    cardd = root / 'data/processed/cardd_yolo'
    split_dir = dict(training='train', validation='val', test='test')
    counts = {}
    for name, d in split_dir.items():
        labels = list((cardd / 'labels' / d).glob('*.txt'))
        counts[name] = (sum(1 for p in (cardd / 'images' / d).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}),
                        sum(len(p.read_text().splitlines()) for p in labels))

    base_dir = _latest(root, 'yolo11n_baseline_val_*', 'metrics.json')
    base = _json(base_dir / 'metrics.json') if base_dir else None
    thr_dir = _latest(root, 'yolo11n_threshold_audit_*', 'summary.csv')
    thr = next((r for r in csv.DictReader((thr_dir / 'summary.csv').open()) if float(r['confidence']) == .25), None) if thr_dir else None
    err_dir = _latest(root, 'yolo11n_error_analysis_*', 'per_image.csv')
    by_count = None
    if err_dir:
        g = defaultdict(lambda: [0, 0])
        for r in csv.DictReader((err_dir / 'per_image.csv').open()):
            k = min(int(r['ground_truth']), 5)
            g[k][0] += int(r['tp']); g[k][1] += int(r['ground_truth'])
        by_count = {('5+' if k == 5 else str(k)): g[k][0] / g[k][1] for k in sorted(g) if g[k][1]}

    audit_dir = _latest(root, 'severity_data3a_audit_*', 'summary.json')
    audit = _json(audit_dir / 'summary.json') if audit_dir else None
    crop_dir = _latest(root, 'severity_crops_*', 'summary.json')
    crops = _json(crop_dir / 'summary.json') if crop_dir else None
    trained = []
    for p in sorted((root / 'outputs/experiments').glob('severity_resnet50_*')):
        m = _json(p / 'val_metrics.json')
        if m:
            parts = p.name.split('_')
            trained.append(dict(variant=parts[2], seed=parts[3][4:], best_epoch=m['best_epoch'], epochs_run=m['epochs_run'],
                                macro_f1=m['validation']['macro_f1']))
    evals = {}
    for split in ['validation', 'test']:
        d = _latest(root, f'severity_eval_{split}_*', 'summary.json')
        if d:
            evals[split] = (d, _json(d / 'summary.json'))
    card = _json(root / 'models/resnet50/model_card.json')

    f3 = lambda v: f'{v:.3f}'
    pct = lambda v: f'{v:.0%}'

    def sev(split, variant, key='macro_f1'):
        return evals[split][1]['results'][f'{variant}->{variant}']['all'][key]['mean']

    def ci(split, key):
        return evals[split][1]['bootstrap']['intervals'][key]

    nodes = []

    def node(id, letter, col, rows, title, sub, detail, facts=(), script=None, output=None, cols=1, pending=False):
        nodes.append(dict(id=id, letter=letter, col=col, cols=cols, row=rows[0], rows=rows[-1] - rows[0] + 1, title=title,
                          sub=sub, detail=detail, facts=[[f[0], str(f[1])] for f in facts if f[1] not in (None, False)], script=script,
                          output=output, pending=pending))

    # Detection track.
    for row, split in enumerate(['training', 'validation', 'test']):
        n, boxes = counts[split]
        node(f'cardd_{split}', 'A', 0, [row], f'CarDD {split}', f'{n:,} images · {boxes:,} boxes',
             f'CarDD {split} split, kept as published. Every annotated damage type is merged into one "damage" class.',
             [('Images', f'{n:,}'), ('Damage boxes', f'{boxes:,}')], output=f'data/processed/cardd_yolo/images/{split_dir[split]}')
    node('convert', 'B', 1, [0, 2], 'COCO → YOLO labels', '1 class: damage',
         'Converts COCO polygons/boxes to YOLO text labels (normalised xywh), one class. Splits stay separate, so no image moves between training, validation and test.',
         [('Classes', '1 (damage)'), ('Config', 'data/processed/cardd_yolo/data.yaml')], script='src/convert_cardd_to_yolo.py', output='data/processed/cardd_yolo')
    node('yolo_train', 'C', 2, [0], 'Train YOLO11n', f"{_yaml_value(run / 'args.yaml', 'epochs') or '?'} epochs · imgsz {_yaml_value(run / 'args.yaml', 'imgsz') or '?'}",
         'Fine-tunes the COCO-pretrained YOLO11n on CarDD training images only. Validation images are scored after each epoch to monitor progress; they are never used for weight updates.',
         [('Base weights', _yaml_value(run / 'args.yaml', 'model')), ('Epochs', _yaml_value(run / 'args.yaml', 'epochs')),
          ('Batch', _yaml_value(run / 'args.yaml', 'batch')), ('Image size', _yaml_value(run / 'args.yaml', 'imgsz')),
          ('Output', 'weights/best.pt')], output=_rel(root, run), cols=2)
    node('yolo_val', 'D', 4, [1], 'Validate detector',
         f"mAP50 {f3(base['metrics/mAP50(B)'])} · recall {pct(thr and float(thr['recall']) or 0)}" if base else 'not run yet',
         'Step 1 re-validates best.pt on the validation split and reproduces the saved metrics. Step 2 audits confidence thresholds; 0.25 is kept. '
         'Step 3 matches predictions to annotations per image: recall falls sharply as the number of damaged areas grows.',
         [('mAP50 (reproduced)', base and f3(base['metrics/mAP50(B)'])),
          ('mAP50–95', base and f3(base['metrics/mAP50-95(B)'])),
          ('Precision @ 0.25', thr and pct(float(thr['precision']))), ('Recall @ 0.25', thr and pct(float(thr['recall']))),
          ('False positives @ 0.25', thr and f"{int(thr['fp']):,}"),
          ('Recall by damages/image', by_count and ' · '.join(f'{k}: {pct(v)}' for k, v in by_count.items()))],
         script='src/validate_yolo_baseline.py · analyze_yolo_thresholds.py · visualize_yolo_errors.py', output=_rel(root, err_dir or thr_dir or base_dir))
    node('yolo_test', 'F', 5, [2], 'Detector test', 'held out · final project',
         'The CarDD test split has not been evaluated. It is reserved for one evaluation of the final, stronger detector, so detector choices cannot be tuned on it.',
         [('Status', 'Not evaluated (by design)')], pending=True)
    node('yolo_result', 'E', 6, [0, 1], 'Detector result',
         f"P {f3(base['metrics/precision(B)'])} · R {f3(base['metrics/recall(B)'])}" if base else 'not run yet',
         'Validation metrics of best.pt (re-validated in step D). Usable for localisation of a single clear damage, but it misses many areas on images with several damages. '
         'Its boxes feed the severity crops (step I) and the Test model tab.',
         [('Precision', base and f3(base['metrics/precision(B)'])), ('Recall', base and f3(base['metrics/recall(B)'])),
          ('mAP50', base and f3(base['metrics/mAP50(B)'])), ('mAP50–95', base and f3(base['metrics/mAP50-95(B)'])),
          ('Validation images', base and base.get('images'))],
         output=_rel(root, run / 'weights/best.pt'))

    # Severity track.
    for i, split in enumerate(['training', 'validation', 'test']):
        c = audit and audit['counts'][split]
        orig = c and sum(v['original'] for v in c.values())
        node(f'data3a_{split}', 'G', 0, [3 + i], f'data3a {split}', f'{orig:,} images · 3 classes' if c else 'not audited yet',
             f'data3a {split} folder with minor / moderate / severe labels, as supplied.',
             [(k.split('-', 1)[1].capitalize(), f"{v['original']} → {v['included']} kept") for k, v in (c or {}).items()],
             output=f'data/data3a/{split}')
    if audit:
        ex = audit['excluded_by_reason']
        inc = {s: sum(v['included'] for v in audit['counts'][s].values()) for s in audit['counts']}
    node('audit', 'H', 1, [3, 5], 'Leakage audit', f"{audit['images']:,} → {audit['included']:,} images" if audit else 'not run yet',
         'Exact (SHA-256) and near (dHash) duplicates are grouped. Copies across splits are dropped from the later split, conflicting labels are dropped, '
         'and validation/test images that also appear in CarDD are excluded so the detector cannot have seen them.',
         [('Kept · train / val / test', audit and f"{inc['training']:,} / {inc['validation']:,} / {inc['test']:,}"),
          ('Same-split duplicates', audit and ex.get('same_split_duplicate_of')), ('Cross-split duplicates', audit and ex.get('cross_split_duplicate_of')),
          ('Label conflicts', audit and ex.get('label_conflict')), ('CarDD overlap', audit and ex.get('cardd_overlap'))],
         script='src/audit_severity_dataset.py', output=_rel(root, audit_dir))
    node('crops', 'I', 2, [3, 5], 'Two input variants', 'whole image | YOLO crop',
         'Whole image: the photo as is. Localized: the union of YOLO boxes (confidence ≥ 0.25) padded 10% per side; when nothing is detected the whole image is used (fallback). '
         'The same frozen detector is applied to every split.',
         [(f'Fallback · {s}', f"{v['fallback']} of {v['images']} ({pct(v['fallback_rate'])})") for s, v in (crops or {}).items()],
         script='src/prepare_severity_crops.py', output=_rel(root, crop_dir))
    seeds = sorted({t['seed'] for t in trained})
    node('sev_train', 'J', 3, [3], 'Train ResNet50 ×6', f"2 inputs × {len(seeds)} seeds" if trained else 'not run yet',
         'ImageNet ResNet50: 5 head-only epochs, then full fine-tuning (AdamW, cosine schedule, label smoothing 0.1). '
         'Only training images update weights; the test split is never loaded by the training script.',
         [(f"{t['variant']} · seed {t['seed']}", f"best epoch {t['best_epoch']} of {t['epochs_run']}") for t in trained],
         script='src/train_severity_classifier.py', output='outputs/experiments/severity_resnet50_*')
    node('sev_val', 'K', 4, [4], 'Select on validation',
         f"F1 whole {f3(sev('validation', 'whole'))} · crop {f3(sev('validation', 'crop'))}" if 'validation' in evals else 'not run yet',
         'After each epoch the model is scored on validation macro-F1; the best epoch is kept and training stops after 7 epochs without improvement. '
         'Because selection used these images, validation scores are optimistic.',
         [(f"{t['variant']} · seed {t['seed']}", f"macro-F1 {f3(t['macro_f1'])}") for t in trained]
         + ([('Mean whole / crop', f"{f3(sev('validation', 'whole'))} / {f3(sev('validation', 'crop'))}")] if 'validation' in evals else []),
         script='src/evaluate_severity.py --split validation', output=_rel(root, evals.get('validation', (None,))[0]))
    if 'test' in evals:
        d = ci('test', 'difference_whole_minus_crop/macro_f1')
    node('sev_test', 'L', 5, [5], 'Test once',
         f"F1 whole {f3(sev('test', 'whole'))} · crop {f3(sev('test', 'crop'))}" if 'test' in evals else 'not run yet',
         'The six selected checkpoints are frozen and run once on the test images. Results are the mean of 3 seeds with a paired bootstrap (2,000 resamples) over images.',
         [('Images', 'test' in evals and evals['test'][1]['images']),
          ('Macro-F1 whole', 'test' in evals and f"{f3(sev('test', 'whole'))} [{f3(ci('test', 'whole->whole/macro_f1')['low'])}, {f3(ci('test', 'whole->whole/macro_f1')['high'])}]"),
          ('Macro-F1 crop', 'test' in evals and f"{f3(sev('test', 'crop'))} [{f3(ci('test', 'crop->crop/macro_f1')['low'])}, {f3(ci('test', 'crop->crop/macro_f1')['high'])}]"),
          ('Accuracy whole / crop', 'test' in evals and f"{sev('test', 'whole', 'accuracy'):.1%} / {sev('test', 'crop', 'accuracy'):.1%}"),
          ('Majority-class accuracy', 'test' in evals and f"{evals['test'][1]['majority_baseline_accuracy']:.1%}")],
         script='src/evaluate_severity.py --split test', output=_rel(root, evals.get('test', (None,))[0]))
    node('sev_result', 'M', 6, [3, 4], 'Severity result',
         f"ΔF1 (crop − whole) {sev('test', 'crop') - sev('test', 'whole'):+.3f}".replace('-', '−') if 'test' in evals else 'not run yet',
         'Localized (crop) input scored lower than the whole image, and the bootstrap interval excludes zero. With the current detector, missed and partial boxes remove context the classifier needs.',
         [('ΔF1 = crop − whole', 'test' in evals and f"{sev('test', 'crop') - sev('test', 'whole'):+.3f} [{-d['high']:+.3f}, {-d['low']:+.3f}]"),
          ('Moderate recall whole / crop', 'test' in evals and f"{f3(sev('test', 'whole', 'recall_02-moderate'))} / {f3(sev('test', 'crop', 'recall_02-moderate'))}")],
         output=_rel(root, evals.get('test', (None,))[0]))
    node('deploy', 'N', 6, [5], 'Dashboard model', 'whole image · seed 813' if card else 'no checkpoint',
         'The Test model tab runs YOLO for boxes and the whole-image ResNet50 for severity. Seed 813 had the best validation macro-F1 of the whole-image seeds; test results were not used to choose it.',
         [('Checkpoint', 'models/resnet50/severity_resnet50_whole_seed813.pt'), ('Selected epoch', card and card['selected_epoch']),
          ('Validation macro-F1', card and f3(card['validation_macro_f1']))], output='models/resnet50/model_card.json')

    edges = [('cardd_training', 'convert'), ('cardd_validation', 'convert'), ('cardd_test', 'convert'), ('convert', 'yolo_train'),
             ('convert', 'yolo_val'), ('convert', 'yolo_test'), ('yolo_train', 'yolo_val', 'best.pt'),
             ('yolo_val', 'yolo_result'), ('yolo_train', 'crops', 'frozen detector'),
             ('data3a_training', 'audit'), ('data3a_validation', 'audit'), ('data3a_test', 'audit'), ('audit', 'crops'),
             ('crops', 'sev_train'), ('crops', 'sev_val'), ('crops', 'sev_test'),
             ('sev_train', 'sev_val', 'each epoch'), ('sev_val', 'sev_test', 'best epochs'), ('sev_test', 'sev_result'), ('sev_val', 'deploy', 'best seed')]
    return dict(columns=COLUMNS, lanes=[dict(track=t, split=s) for t, s in LANES], nodes=nodes,
                edges=[dict(source=e[0], target=e[1], label=e[2] if len(e) > 2 else '') for e in edges])
