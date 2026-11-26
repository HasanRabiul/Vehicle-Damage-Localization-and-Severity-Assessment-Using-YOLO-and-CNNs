"""A-to-Z process map: every step of detection and severity, placed by data split, with numbers read from saved outputs.

Layout: columns are process stages, rows are split lanes (training / validation / test) for each track. A missing
output leaves its node marked 'not run yet' instead of inventing a value.
"""
import json

from findings import findings

COLUMNS = ['Data', 'Prepare', 'Build inputs', 'Train', 'Validate & select', 'Test (once)', 'Result', 'Analyse & use']
LANES = [('detection', 'training'), ('detection', 'validation'), ('detection', 'test'),
         ('severity', 'training'), ('severity', 'validation'), ('severity', 'test')]


def _json(path):
    return json.loads(path.read_text()) if path and path.exists() else None


def _epochs(root, name):
    p = root / 'runs/detect/runs/damage_detection' / name / 'results.csv'
    return sum(1 for _ in p.open()) - 1 if p.exists() else None


def pipeline(root):
    F = findings(root)
    cardd = root / 'data/processed/cardd_yolo'
    split_dir = dict(training='train', validation='val', test='test')
    counts = {}
    for name, d in split_dir.items():
        labels = list((cardd / 'labels' / d).glob('*.txt'))
        counts[name] = (sum(1 for p in (cardd / 'images' / d).glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}),
                        sum(len(p.read_text().splitlines()) for p in labels))
    audit_dirs = sorted((root / 'outputs/experiments').glob('severity_data3a_audit_*'))
    audit = _json(audit_dirs[-1] / 'summary.json') if audit_dirs else None
    cfg = F['final_configs']
    crops = _json(root / cfg['localized']['crops'] / 'summary.json') if cfg else None

    f3 = lambda v: f'{v:.3f}'
    pct = lambda v: f'{v:.0%}'
    sgn = lambda v: f'{v:+.3f}'.replace('-', '−')
    rng = lambda c: f"[{sgn(c['low'])}, {sgn(c['high'])}]"

    nodes = []

    def node(id, letter, col, rows, title, sub, detail, facts=(), script=None, output=None, cols=1, pending=False):
        nodes.append(dict(id=id, letter=letter, col=col, cols=cols, row=rows[0], rows=rows[-1] - rows[0] + 1, title=title,
                          sub=sub, detail=detail, facts=[[f[0], str(f[1])] for f in facts if f[1] not in (None, False, '')], script=script,
                          output=output, pending=pending))

    # Detection track.
    for row, split in enumerate(['training', 'validation', 'test']):
        n, boxes = counts[split]
        node(f'cardd_{split}', 'A', 0, [row], f'CarDD {split}', f'{n:,} images · {boxes:,} boxes',
             f'CarDD {split} split, kept as published. Boxes carry one of six damage types (dent, scratch, crack, glass shatter, lamp broken, tire flat).',
             [('Images', f'{n:,}'), ('Damage boxes', f'{boxes:,}')], output=f'data/processed/cardd_yolo/images/{split_dir[split]}')
    node('convert', 'B', 1, [0, 2], 'COCO → YOLO labels', '1 class · plus a 6-type copy',
         'Converts the COCO boxes to YOLO text labels. The pipeline uses one merged "damage" class, because the severity stage only needs where the damage is. '
         'A second copy keeps the six damage types for the six-class study (step G). Splits stay separate.',
         [('Pipeline classes', '1 (damage)'), ('Six-class copy', 'data/processed/cardd_yolo6')],
         script='src/convert_cardd_to_yolo.py · convert_cardd_to_yolo6.py', output='data/processed/cardd_yolo')
    trained = [(n, _epochs(root, n)) for n in ['yolo11n_initial', 'yolo11n_e100', 'yolo11s_e100', 'yolo11s_e100_repeat', 'yolo11s_6cls_e100']]
    node('yolo_train', 'C', 2, [0], 'Train detectors', 'YOLO11n 10 / 100 ep · YOLO11s 100 ep · repeat · 6-class',
         'COCO-pretrained YOLO11 models fine-tuned on CarDD training images only (imgsz 640, batch 8, early stopping patience 20). '
         'Validation is scored after each epoch to keep best.pt; it never updates weights. An identical YOLO11s repeat measures run-to-run variation.',
         [(n, f'{e} epochs run') for n, e in trained if e], output='runs/detect/runs/damage_detection', cols=2)
    sel, test = F.get('detector_selection'), F.get('detector_test')
    chosen = sel and next(m for m in sel['models'] if m['model'] == sel['chosen'])
    node('yolo_val', 'D', 4, [1], 'Select detector', f"{sel['chosen']} · conf {sel['confidence']:.2f} · F1 {f3(chosen['f1'])}" if sel else 'not run yet',
         'Every candidate is run once on the 810 validation images. For each model the confidence threshold with the best F1 (IoU ≥ 0.5, one-to-one matching) is kept; '
         'a larger model must beat YOLO11n-100 by at least 0.01 F1. The chosen detector and threshold are frozen.',
         [(m['model'], f"conf {m['confidence']:.2f} · P {f3(m['precision'])} · R {f3(m['recall'])} · F1 {f3(m['f1'])}") for m in (sel or {}).get('models', [])]
         + [('Rule', sel and sel['rule'])], script='src/select_detector.py', output=sel and sel['run'])
    t = test and {r['model']: r for r in test['results']}
    c = t and next(r for r in t.values() if r['role'] == 'chosen')
    node('yolo_test', 'E', 5, [2], 'Detector test (once)', f"mAP50 {f3(c['mAP50'])} · recall {f3(c['recall'])}" if c else 'not run yet',
         'The frozen detector and the round-1 YOLO11n are evaluated once on the 374 CarDD test images. Nothing is chosen from this result.',
         [(f"{k} ({r['role']})", f"mAP50 {f3(r['mAP50'])} · R {f3(r['recall'])} · F1 {f3(r['f1'])} · no detection {r['no_detection_images']}") for k, r in (t or {}).items()],
         script='src/evaluate_detector_test.py', output=test and test['run'], pending=not test)
    ref = t and next((r for r in t.values() if r['role'] == 'reference'), None)
    node('yolo_result', 'F', 6, [0, 1], 'Detector result', f"test mAP50 {f3(ref['mAP50'])} → {f3(c['mAP50'])}" if c and ref else 'not run yet',
         'Longer training and the larger YOLO11s raise test mAP50 and recall clearly over round 1, and images with no detection drop sharply. '
         'Recall still falls as the number of damaged areas per image grows.',
         [('Recall', ref and f"{f3(ref['recall'])} → {f3(c['recall'])}"), ('F1', ref and f"{f3(ref['f1'])} → {f3(c['f1'])}"),
          ('Images with no detection', ref and f"{ref['no_detection_images']} → {c['no_detection_images']}"),
          ('Recall by damages/image', c and ' · '.join(f'{k}: {pct(v)}' for k, v in c['recall_by_damages'].items()))],
         output=cfg and cfg['detector']['checkpoint'])
    sv, st = F.get('six_class_val'), F.get('six_class_test')
    six = lambda s, k='yolo11s_6cls_e100': s and s['results'].get(k)
    gap = sv and abs(six(sv, 'yolo11s_e100')['mAP50'] - six(sv, 'yolo11s_e100_repeat')['mAP50'])
    node('six', 'G', 7, [0, 2], 'Six-class study', f"test recall {f3(six(st, 'yolo11s_e100')['best']['recall'])} → {f3(six(st)['best']['recall'])}" if st else 'not run yet',
         'The same YOLO11s trained with the six damage types kept. Scored class-agnostically (types ignored) against the one-class models on validation, then once on test. '
         'It is a side study: the frozen pipeline detector is not replaced.',
         [('Val mAP50 · 1-class / repeat / 6-class', sv and f"{f3(six(sv, 'yolo11s_e100')['mAP50'])} / {f3(six(sv, 'yolo11s_e100_repeat')['mAP50'])} / {f3(six(sv)['mAP50'])}"),
          ('Run-to-run gap (identical 1-class runs)', gap and f3(gap)),
          ('Test mAP50 · 1-class / 6-class', st and f"{f3(six(st, 'yolo11s_e100')['mAP50'])} / {f3(six(st)['mAP50'])}"),
          ('Test AP50 by type', st and ' · '.join(f"{k} {v['AP50']:.2f}" for k, v in six(st)['per_type'].items()))],
         script='src/compare_six_class_detector.py', output=st and st['run'], pending=not st)

    # Severity track.
    for i, split in enumerate(['training', 'validation', 'test']):
        cn = audit and audit['counts'][split]
        orig = cn and sum(v['original'] for v in cn.values())
        node(f'data3a_{split}', 'H', 0, [3 + i], f'data3a {split}', f'{orig:,} images · 3 classes' if cn else 'not audited yet',
             f'data3a {split} folder with minor / moderate / severe labels, as supplied.',
             [(k.split('-', 1)[1].capitalize(), f"{v['original']} → {v['included']} kept") for k, v in (cn or {}).items()],
             output=f'data/data3a/{split}')
    if audit:
        ex = audit['excluded_by_reason']
        inc = {s: sum(v['included'] for v in audit['counts'][s].values()) for s in audit['counts']}
    node('audit', 'I', 1, [3, 5], 'Leakage audit', f"{audit['images']:,} → {audit['included']:,} images" if audit else 'not run yet',
         'Exact (SHA-256) and near (dHash) duplicates are grouped. Copies across splits are dropped from the later split, conflicting labels are dropped, '
         'and validation/test images that also appear in CarDD are excluded so the detector cannot have seen them.',
         [('Kept · train / val / test', audit and f"{inc['training']:,} / {inc['validation']:,} / {inc['test']:,}"),
          ('Same-split duplicates', audit and ex.get('same_split_duplicate_of')), ('Cross-split duplicates', audit and ex.get('cross_split_duplicate_of')),
          ('Label conflicts', audit and ex.get('label_conflict')), ('CarDD overlap', audit and ex.get('cardd_overlap'))],
         script='src/audit_severity_dataset.py', output=audit_dirs and str(audit_dirs[-1].relative_to(root)))
    M = F['models']
    node('crops', 'J', 2, [3, 5], 'Whole vs localized input', f"localized: YOLO11s ≥ {M['confidence']:.2f}, pad {M['pad']:.0%}" if cfg else 'not run yet',
         'Whole image: the photo as is. Localized: the union of the frozen detector\'s boxes, padded per side and clipped; when nothing is detected the whole image is used (fallback). '
         'The same frozen detector is applied to every split.',
         [(f'Fallback · {s}', f"{v['fallback']} of {v['images']} ({pct(v['fallback_rate'])})") for s, v in (crops or {}).items()]
         + [('Crop variants built', len(F['crop_variants']))], script='src/prepare_severity_crops.py', output=cfg and cfg['localized']['crops'])
    node('sev_train', 'K', 3, [3], 'Train ResNet50', 'whole ×3 seeds · crop variants',
         'ImageNet ResNet50: 5 head-only epochs, then full fine-tuning (AdamW, cosine schedule, label smoothing 0.1), seeds 812–814. '
         'Only training images update weights; the test split is never loaded by the training script.',
         [(f"crop · {v['detector']} ≥ {v['confidence']} · pad {v['pad']:.0%}", ' · '.join(f'seed {k}' for k in v['seeds'])) for v in F['crop_variants']]
         + [('whole image', 'seeds 812 · 813 · 814 (round 1, reused)')], script='src/train_severity_classifier.py', output='outputs/experiments/severity_resnet50_*')
    chosen_v = next((v for v in F['crop_variants'] if v['chosen']), None)
    sval = F.get('severity_validation')
    node('sev_val', 'L', 4, [4], 'Select & freeze', f"crop variant F1 {f3(chosen_v['selection_mean'])} · frozen" if chosen_v else 'not run yet',
         'Each training run keeps its best epoch by validation macro-F1. The crop variant with the best mean validation F1 (seeds 812–813) is chosen, '
         'and the whole configuration is written to final_configs.json before the test run. Validation scores are optimistic.',
         [(f"{v['detector']} ≥ {v['confidence']} · pad {v['pad']:.0%}{' (round 1)' if v['round1'] else ''}{' ← chosen' if v['chosen'] else ''}",
           f"mean F1 {f3(v['selection_mean'])}" if v['selection_mean'] else '') for v in F['crop_variants']]
         + [('Validation F1 whole / localized', sval and f"{f3(sval['variants']['whole']['macro_f1'])} / {f3(sval['variants']['crop']['macro_f1'])}"),
            ('Frozen at', cfg and cfg['frozen_utc'][:16].replace('T', ' ') + ' UTC')],
         script='src/evaluate_severity.py --split validation', output='final_configs.json')
    tests = F['severity_test']
    fin = tests[-1] if tests and tests[-1].get('final_comparison') else None
    r1 = tests[0] if tests else None
    node('sev_test', 'M', 5, [5], 'Test once (final)',
         f"F1 whole {f3(fin['variants']['whole']['macro_f1'])} · localized {f3(fin['variants']['crop']['macro_f1'])}" if fin else 'not run yet',
         'The six frozen checkpoints run once on the 245 test images (second and final use; round 1 was the first). Mean of 3 seeds, paired bootstrap over images (2,000 resamples).',
         [('Macro-F1 whole', fin and f"{f3(fin['variants']['whole']['macro_f1'])} {rng(fin['variants']['whole']['ci'])}"),
          ('Macro-F1 localized', fin and f"{f3(fin['variants']['crop']['macro_f1'])} {rng(fin['variants']['crop']['ci'])}"),
          ('Moderate recall whole / localized', fin and f"{f3(fin['variants']['whole']['recall']['02-moderate'])} / {f3(fin['variants']['crop']['recall']['02-moderate'])}"),
          ('Majority-class accuracy', fin and f"{fin['majority_accuracy']:.1%}")],
         script='src/evaluate_severity.py --split test · bootstrap_final_test.py', output=fin and fin['run'], pending=not fin)
    node('sev_result', 'N', 6, [3, 4], 'Severity result', f"ΔF1 {sgn(fin['delta_f1'])} {rng(fin['delta_f1_ci'])}" if fin else 'not run yet',
         'With the stronger detector, localized input no longer hurts severity classification (round 1 did, significantly), but it does not help either: '
         'both pre-registered measures have intervals that include zero.',
         [('ΔF1 = localized − whole', fin and f"{sgn(fin['delta_f1'])} {rng(fin['delta_f1_ci'])}"),
          ('ΔR moderate', fin and f"{sgn(fin['delta_r_moderate'])} {rng(fin['delta_r_moderate_ci'])}"),
          ('Round 1 ΔF1 (YOLO11n crops)', r1 and r1 is not fin and f"{sgn(r1['delta_f1'])} {rng(r1['delta_f1_ci'])}")],
         output=fin and fin['run'] + '/final_comparison.json')
    ea = F.get('error_analysis')
    g = ea and ea['gradcam']
    sub = ea and {s['subset']: s for s in ea['subsets']}
    node('analysis', 'O', 7, [5], 'Error analysis', f"Grad-CAM ratio {g['mean_ratio']:.2f} · tight crops hurt" if ea else 'not run yet',
         'Explains the fixed test predictions; nothing is chosen. Per-class metrics, detected vs fallback images, crop size, where moderate images go, '
         'and Grad-CAM of the whole-image models: how much attention falls inside the YOLO boxes relative to their area.',
         [('Grad-CAM attention ratio (mean · >1 share)', g and f"{g['mean_ratio']:.2f} · {pct(g['fraction_above_1'])} of {g['images']} images"),
          ('Crop − whole F1 · crop < 1/3 image', sub and sgn(sub['crop < 1/3 of image']['crop_minus_whole_f1'])),
          ('Crop − whole F1 · crop ≥ 2/3 image', sub and sgn(sub['crop >= 2/3 of image']['crop_minus_whole_f1'])),
          ('Moderate → minor / moderate / severe (whole)', ea and ' / '.join(pct(v) for v in ea['moderate_predicted_as']['whole'].values()))],
         script='src/analyze_final_errors.py', output=ea and ea['run'], pending=not ea)
    node('deploy', 'P', 7, [3, 4], 'Dashboard models', 'YOLO11s + whole & localized ResNet50',
         'The Test model tab runs the frozen detector, builds the localized crop with the frozen settings, and classifies both inputs. '
         'For each input the seed with the best validation macro-F1 is used; test results were not used to choose them.',
         [('Whole input', M['whole'] and f"seed {M['whole']['seed']} · val F1 {f3(M['whole']['validation_macro_f1'])}"),
          ('Localized input', M['crop'] and f"seed {M['crop']['seed']} · val F1 {f3(M['crop']['validation_macro_f1'])}"),
          ('Detector', M['detector'])], output='webapp/app.py')

    edges = [('cardd_training', 'convert'), ('cardd_validation', 'convert'), ('cardd_test', 'convert'), ('convert', 'yolo_train'),
             ('convert', 'yolo_val'), ('convert', 'yolo_test'), ('yolo_train', 'yolo_val', 'best.pt'), ('yolo_val', 'yolo_test', 'frozen'),
             ('yolo_test', 'yolo_result'), ('yolo_result', 'six', 'compared'), ('yolo_train', 'crops', 'frozen YOLO11s'),
             ('data3a_training', 'audit'), ('data3a_validation', 'audit'), ('data3a_test', 'audit'), ('audit', 'crops'),
             ('crops', 'sev_train'), ('crops', 'sev_val'), ('crops', 'sev_test'),
             ('sev_train', 'sev_val', 'each epoch'), ('sev_val', 'sev_test', 'frozen'), ('sev_test', 'sev_result'), ('sev_test', 'analysis'),
             ('sev_result', 'deploy')]
    return dict(columns=COLUMNS, lanes=[dict(track=t, split=s) for t, s in LANES], nodes=nodes,
                edges=[dict(source=e[0], target=e[1], label=e[2] if len(e) > 2 else '') for e in edges])
