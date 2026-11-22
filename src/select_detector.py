"""Choose the detector and its confidence threshold on CarDD validation only. Test images are not read."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

RUNS = ROOT/'runs/detect/runs/damage_detection'
CANDIDATES = {'yolo11n_e10': RUNS/'yolo11n_initial/weights/best.pt',
              'yolo11n_e100': RUNS/'yolo11n_e100/weights/best.pt',
              'yolo11s_e100': RUNS/'yolo11s_e100/weights/best.pt'}
DATA = ROOT/'data/processed/cardd_yolo'
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MATCH_IOU = 0.5
SMALL_MODEL, MIN_GAIN = 'yolo11n_e100', 0.01  # a larger model must beat the small one by at least this F1


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def ground_truth(image, w, h):
    lab = DATA/'labels/val'/(image.stem + '.txt')
    rows = [list(map(float, l.split()[1:])) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
    return np.array([[(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h] for x, y, bw, bh in rows]).reshape(-1, 4)


def true_positives(boxes, scores, gt):
    """Greedy one-to-one matching in descending score order at IoU >= MATCH_IOU."""
    if not len(boxes) or not len(gt):
        return 0
    used, tp = set(), 0
    for row in iou(boxes[np.argsort(-scores)], gt):
        j = next((j for j in np.argsort(-row) if row[j] >= MATCH_IOU and j not in used), None)
        if j is not None:
            used.add(j); tp += 1
    return tp


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = ROOT/'outputs/experiments'/('detector_selection_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    images = sorted(p for p in (DATA/'images/val').iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    rows = []
    for name, ckpt in CANDIDATES.items():
        cache, model = [], YOLO(str(ckpt))
        for r in model.predict(str(DATA/'images/val'), conf=min(THRESHOLDS), iou=0.7, imgsz=640, batch=16,
                        device='mps', stream=True, verbose=False):
            h, w = r.orig_shape
            cache.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), ground_truth(Path(r.path), w, h)))
        del model
        torch.mps.empty_cache()
        assert len(cache) == len(images), f'{name}: predicted {len(cache)} of {len(images)} images'    
        for t in THRESHOLDS:
            tp = fp = fn = empty = 0
            by_count = defaultdict(lambda: [0, 0])
            for boxes, scores, gt in cache:
                keep = scores >= t
                k = true_positives(boxes[keep], scores[keep], gt)
                tp += k; fp += int(keep.sum()) - k; fn += len(gt) - k; empty += int(not keep.any())
                by_count[min(len(gt), 5)][0] += k; by_count[min(len(gt), 5)][1] += len(gt)
            p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append(dict(model=name, confidence=t, tp=tp, fp=fp, fn=fn, precision=round(p, 4), recall=round(rc, 4),
                             f1=round(2*p*rc / max(p + rc, 1e-9), 4), no_detection_images=empty,
                             **{f'recall_gt_{k if k < 5 else "5plus"}': round(v[0]/v[1], 4) for k, v in sorted(by_count.items()) if v[1]}))
        print(f'{name}: done')

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out/'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    best = {m: max((r for r in rows if r['model'] == m), key=lambda r: (r['f1'], -r['confidence'])) for m in CANDIDATES}
    top = max(best.values(), key=lambda r: r['f1'])
    chosen = best[SMALL_MODEL] if top['f1'] - best[SMALL_MODEL]['f1'] < MIN_GAIN else top
    selection = dict(rule=f'Per model, the threshold with the highest validation F1 (ties: lower threshold). Choose the highest-F1 model, '
                          f'unless it beats {SMALL_MODEL} by less than {MIN_GAIN} F1, then keep {SMALL_MODEL}.',
                     best_per_model=best, chosen=chosen, checkpoint=str(CANDIDATES[chosen['model']].relative_to(ROOT)))
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    (out/'config.json').write_text(json.dumps(dict(split='CarDD validation', images=len(images), thresholds=THRESHOLDS, match_iou=MATCH_IOU,
        nms_iou=0.7, imgsz=640, candidates={k: dict(path=str(v.relative_to(ROOT)), sha256=sha256(v)) for k, v in CANDIDATES.items()},
        note='Test split not read. best.pt epochs were also chosen on validation, so these numbers are optimistic.'), indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key in zip(axes, ['precision', 'recall', 'f1']):
        for m in CANDIDATES:
            r = [x for x in rows if x['model'] == m]
            ax.plot([x['confidence'] for x in r], [x[key] for x in r], 'o-', label=m)
        ax.set(xlabel='Confidence threshold', title=key.capitalize(), ylim=(0, 1)); ax.grid(alpha=.3)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out/'threshold_selection.png', dpi=180); plt.close(fig)

    print(f'\n{"model":14} {"conf":>5} {"P":>6} {"R":>6} {"F1":>6} {"FP":>5} {"no-det":>6}  recall by damages/image (1,2,3,4,5+)')
    for r in rows:
        bc = ' '.join(f'{r.get(f"recall_gt_{k}", 0):.2f}' for k in ['1', '2', '3', '4', '5plus'])
        print(f'{r["model"]:14} {r["confidence"]:5.2f} {r["precision"]:6.3f} {r["recall"]:6.3f} {r["f1"]:6.3f} {r["fp"]:5d} {r["no_detection_images"]:6d}  {bc}')
    print('\nBest per model:', {m: (b['confidence'], b['f1']) for m, b in best.items()})
    print('CHOSEN:', chosen['model'], 'at confidence', chosen['confidence'], '->', selection['checkpoint'])
    print('Saved to', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
