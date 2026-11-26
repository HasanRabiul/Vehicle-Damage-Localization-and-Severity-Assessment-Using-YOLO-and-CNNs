"""Final project findings read from saved outputs: detector selection and test, the six-class study, localization
variants, severity round 1 vs final test, and the error analysis. Missing outputs give None; nothing is recomputed."""
import csv
import json
import re
from collections import defaultdict

RUNS = 'runs/detect/runs/damage_detection'
CURVES = [('yolo11n_initial', 'YOLO11n · 10 epochs (round 1)'), ('yolo11n_e100', 'YOLO11n · 100 epochs'),
          ('yolo11s_e100', 'YOLO11s · 100 epochs (chosen)'), ('yolo11s_e100_repeat', 'YOLO11s · identical repeat')]


def _json(p):
    return json.loads(p.read_text()) if p and p.exists() else None


def _dirs(root, pattern, file):
    return sorted(p for p in (root / 'outputs/experiments').glob(pattern) if (p / file).exists())


def _rel(root, p):
    return str(p.relative_to(root)) if p else None


def _curve(root, name):
    p = root / RUNS / name / 'results.csv'
    if not p.exists():
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(p.open())]
    return dict(epoch=[int(float(r['epoch'])) for r in rows], map50=[float(r['metrics/mAP50(B)']) for r in rows])


def _severity_run(root, d):
    """Mean-of-seeds test (or validation) metrics for whole and crop input from one severity_eval_* folder."""
    s = _json(d / 'summary.json')
    ci = s.get('bootstrap', {}).get('intervals', {})
    res = lambda v, k: s['results'][f'{v}->{v}']['all'][k]['mean']
    out = dict(run=_rel(root, d), images=s['images'], majority_accuracy=s['majority_baseline_accuracy'],
               variants={v: dict(macro_f1=res(v, 'macro_f1'), accuracy=res(v, 'accuracy'),
                                 recall={c: res(v, f'recall_{c}') for c in ['01-minor', '02-moderate', '03-severe']},
                                 ci=ci.get(f'{v}->{v}/macro_f1')) for v in ['whole', 'crop']})
    out['delta_f1'] = out['variants']['crop']['macro_f1'] - out['variants']['whole']['macro_f1']
    out['delta_r_moderate'] = out['variants']['crop']['recall']['02-moderate'] - out['variants']['whole']['recall']['02-moderate']
    d_ci = ci.get('difference_whole_minus_crop/macro_f1')
    out['delta_f1_ci'] = d_ci and dict(low=-d_ci['high'], high=-d_ci['low'])  # stored as whole minus crop
    final = _json(d / 'final_comparison.json')
    if final:
        out['final_comparison'] = final
        out['delta_f1_ci'] = dict(low=final['delta_F1']['ci95_low'], high=final['delta_F1']['ci95_high'])
        out['delta_r_moderate_ci'] = dict(low=final['delta_R_moderate']['ci95_low'], high=final['delta_R_moderate']['ci95_high'])
    return out


def crop_variants(root):
    """Localized-input variants compared on validation: crop folder -> detector, threshold, padding, seed scores."""
    by_crops = defaultdict(dict)
    for d in sorted((root / 'outputs/experiments').glob('severity_resnet50_crop_seed*')):
        c, m = _json(d / 'config.json'), _json(d / 'val_metrics.json')
        if c and m and c.get('crops'):
            by_crops[str(c['crops']).rsplit('/', 1)[0]][int(d.name.split('_')[3][4:])] = m['validation']['macro_f1']
    out = []
    for folder, seeds in by_crops.items():
        cfg, summ = _json(root / folder / 'config.json'), _json(root / folder / 'summary.json')
        if not cfg:
            continue
        pad = re.search(r'padded (\d+)%', cfg.get('crop', ''))
        sel = [seeds[s] for s in (812, 813) if s in seeds]
        out.append(dict(crops=folder, detector=cfg['checkpoint'].split('/')[-3], confidence=cfg['conf'], pad=int(pad.group(1)) / 100 if pad else None,
                        seeds={str(k): v for k, v in sorted(seeds.items())}, selection_mean=sum(sel) / len(sel) if len(sel) == 2 else None,
                        validation_fallback_rate=summ and summ['validation']['fallback_rate'],
                        validation_median_crop_area=summ and summ['validation']['median_crop_area_fraction']))
    return sorted(out, key=lambda r: r['crops'])


def dashboard_models(root):
    """Checkpoints used by the Test model tab: the frozen detector, the six-class detector and the best-validation seed of each input."""
    cfg = _json(root / 'final_configs.json')
    tests = _dirs(root, 'severity_eval_test_*', 'config.json')
    runs = _json(tests[-1] / 'config.json')['runs'] if tests else {}
    best = {}
    for v in ['whole', 'crop']:
        scored = [(m['validation']['macro_f1'], k, r) for k, r in runs.items() if k.startswith(v)
                  for m in [_json(root / r / 'val_metrics.json')] if m]
        if scored:
            f1, k, r = max(scored)
            best[v] = dict(checkpoint=f'{r}/best.pt', seed=int(k.split('seed')[1]), validation_macro_f1=f1)
    crops_cfg = cfg and _json(root / cfg['localized']['crops'] / 'config.json')
    pad = crops_cfg and re.search(r'padded (\d+)%', crops_cfg.get('crop', ''))
    return dict(detector=cfg and cfg['detector']['checkpoint'], confidence=cfg and cfg['detector']['confidence'],
                pad=int(pad.group(1)) / 100 if pad else None, types_detector=f'{RUNS}/yolo11s_6cls_e100/weights/best.pt',
                whole=best.get('whole'), crop=best.get('crop'))


def findings(root):
    out = dict(final_configs=_json(root / 'final_configs.json'))
    out['curves'] = [dict(name=n, label=l, **c) for n, l in CURVES if (c := _curve(root, n))]

    sel = _dirs(root, 'detector_selection_*', 'selection.json')
    if sel:
        s = _json(sel[-1] / 'selection.json')
        out['detector_selection'] = dict(run=_rel(root, sel[-1]), rule=s['rule'], chosen=s['chosen']['model'],
                                         confidence=s['chosen']['confidence'], models=list(s['best_per_model'].values()))
        if (sel[-1] / 'summary.csv').exists():
            out['detector_selection']['sweep'] = [{k: v if k == 'model' else float(v) for k, v in r.items()}
                                                  for r in csv.DictReader((sel[-1] / 'summary.csv').open())]
    test = _dirs(root, 'detector_test_*', 'summary.json')
    if test:
        t = _json(test[-1] / 'summary.json')
        out['detector_test'] = dict(run=_rel(root, test[-1]), images=t['images'], results=[dict(model=k, **v) for k, v in t['results'].items()])

    for split in ['val', 'test']:
        d = _dirs(root, f'six_class_{split}_*', 'summary.json')
        if d:
            s = _json(d[-1] / 'summary.json')
            out[f'six_class_{split}'] = dict(run=_rel(root, d[-1]), images=s['images'], types=s['types'], results=s['results'])

    out['crop_variants'] = crop_variants(root)
    final_crops = out['final_configs'] and out['final_configs']['localized']['crops']
    for v in out['crop_variants']:
        v['chosen'] = v['crops'] == final_crops
        v['round1'] = v['detector'] == 'yolo11n_initial'

    out['severity_test'] = [_severity_run(root, d) for d in _dirs(root, 'severity_eval_test_*', 'summary.json')]
    final = [d for d in _dirs(root, 'severity_eval_test_*', 'final_comparison.json') if (d / 'predictions.csv').exists()]
    if final:
        # Per-image test labels and the six final predictions as class indices (no paths), for the metrics guide.
        rows = list(csv.DictReader((final[-1] / 'predictions.csv').open()))
        labels = sorted({r['label'] for r in rows})
        cols = [c for c in rows[0] if '_seed' in c]
        out['severity_predictions'] = dict(run=_rel(root, final[-1]), labels=[l.split('-', 1)[1] for l in labels],
                                           y=[labels.index(r['label']) for r in rows],
                                           **{v: [[labels.index(r[c]) for r in rows] for c in cols if c.startswith(v)] for v in ['whole', 'crop']})
    val = _dirs(root, 'severity_eval_validation_*', 'summary.json')
    out['severity_validation'] = _severity_run(root, val[-1]) if val else None

    err = _dirs(root, 'severity_error_analysis_*', 'summary.json')
    if err:
        out['error_analysis'] = dict(run=_rel(root, err[-1]), **_json(err[-1] / 'summary.json'),
                                     figures=[_rel(root, err[-1] / f) for f in ['gradcam.png', 'moderate_errors.png'] if (err[-1] / f).exists()])
    out['models'] = dashboard_models(root)
    return out
