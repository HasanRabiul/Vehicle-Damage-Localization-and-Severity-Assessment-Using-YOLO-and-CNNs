"""Six-class vs one-class YOLO11s on CarDD: class-agnostic detection (damage types ignored) and per-type results.
The frozen pipeline detector in final_configs.json is not changed by anything here.
--split val: each model's confidence threshold is chosen by F1, as in select_detector.py.
--split test: run once with the validation thresholds; nothing is chosen."""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from select_detector import DATA, MATCH_IOU, ROOT, RUNS, THRESHOLDS, iou, sha256

DATA6 = ROOT/'data/processed/cardd_yolo6'
SIX = 'yolo11s_6cls_e100'
MODELS = {'yolo11s_e100': RUNS/'yolo11s_e100/weights/best.pt',
          'yolo11s_e100_repeat': RUNS/'yolo11s_e100_repeat/weights/best.pt',
          SIX: RUNS/f'{SIX}/weights/best.pt'}
TEST_MODELS = ['yolo11s_e100', SIX]
_names = yaml.safe_load((DATA6/'data.yaml').read_text())['names']
TYPES = [_names[k] for k in sorted(_names)]
KEY = lambda t: 'recall_' + t.replace(' ', '_')


def ground_truth(split, image, w, h):
    lab = DATA6/'labels'/split/(image.stem + '.txt')
    rows = [list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()]
    boxes = np.array([[(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h] for _, x, y, bw, bh in rows]).reshape(-1, 4)
    return boxes, np.array([int(r[0]) for r in rows], dtype=int)


def matched(boxes, scores, gt):
    """Greedy one-to-one matching as in select_detector.true_positives; True for each ground-truth box that was found."""
    hit = np.zeros(len(gt), bool)
    if len(boxes) and len(gt):
        for row in iou(boxes[np.argsort(-scores)], gt):
            j = next((j for j in np.argsort(-row) if row[j] >= MATCH_IOU and not hit[j]), None)
            if j is not None:
                hit[j] = True
    return hit


def score(cache, t, name):
    tp = fp = empty = 0
    found, total = np.zeros(len(TYPES)), np.zeros(len(TYPES))
    for boxes, scores, gt, cls in cache:
        keep = scores >= t
        hit = matched(boxes[keep], scores[keep], gt)
        tp += int(hit.sum()); fp += int(keep.sum()) - int(hit.sum()); empty += int(not keep.any())
        np.add.at(found, cls, hit.astype(int)); np.add.at(total, cls, 1)
    fn = int(total.sum()) - tp
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(model=name, confidence=t, precision=round(p, 4), recall=round(r, 4), f1=round(2*p*r / max(p + r, 1e-9), 4),
                tp=tp, fp=fp, fn=fn, no_detection_images=empty,
                **{KEY(k): round(float(found[i] / total[i]), 4) for i, k in enumerate(TYPES)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], required=True)
    split = ap.parse_args().split
    exp = ROOT/'outputs/experiments'
    if split == 'test':
        if sorted(exp.glob('six_class_test_*')):
            sys.exit('A six-class test evaluation already exists; the CarDD test split is used once for it.')
        val = json.loads((sorted(exp.glob('six_class_val_*'))[-1]/'summary.json').read_text())
        models = {k: MODELS[k] for k in TEST_MODELS}
        thresholds = {k: [val['results'][k]['best']['confidence']] for k in models}
    else:
        models, thresholds = MODELS, {k: THRESHOLDS for k in MODELS}
    out = exp/(f'six_class_{split}_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    n_images = len([p for p in (DATA6/'images'/split).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    rows, results = [], {}
    for name, ckpt in models.items():
        model = YOLO(str(ckpt))
        va = dict(split=split, imgsz=640, batch=16, device='mps', conf=0.001, iou=0.7, project=str(out), verbose=False)
        # Class-agnostic mAP: the six types merged into one, the same task the one-class models were trained on.
        m = (model.val(data=str(DATA6/'data.yaml'), single_cls=True, plots=False, name=f'{name}_agnostic', **va) if name == SIX
             else model.val(data=str(DATA/'data.yaml'), plots=False, name=f'{name}_agnostic', **va))
        res = dict(checkpoint=str(ckpt.relative_to(ROOT)), sha256=sha256(ckpt),
                   mAP50=round(float(m.box.map50), 4), mAP50_95=round(float(m.box.map), 4))
        if name == SIX:
            t = model.val(data=str(DATA6/'data.yaml'), plots=True, name=f'{name}_per_type', **va)
            res['per_type'] = {TYPES[int(c)]: dict(AP50=round(float(t.box.ap50[i]), 4), AP50_95=round(float(t.box.ap[i]), 4))
                               for i, c in enumerate(t.box.ap_class_index)}
            res['mean_over_types_mAP50'] = round(float(t.box.map50), 4)
        cache = []
        for r in model.predict(str(DATA6/'images'/split), conf=min(thresholds[name]), iou=0.7, imgsz=640, batch=16,
                               device='mps', agnostic_nms=True, stream=True, verbose=False):
            h, w = r.orig_shape
            cache.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), *ground_truth(split, Path(r.path), w, h)))
        del model
        torch.mps.empty_cache()
        assert len(cache) == n_images, f'{name}: predicted {len(cache)} of {n_images} images'
        mine = [score(cache, t, name) for t in thresholds[name]]
        rows += mine
        res['best'] = max(mine, key=lambda r: (r['f1'], -r['confidence']))
        results[name] = res
        print(f'{name}: done')

    with (out/'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    note = ('Thresholds chosen on validation by F1 (ties: lower threshold).' if split == 'val' else
            'Evaluated once with the validation thresholds; nothing chosen.') + \
           ' Class-agnostic: a predicted box counts if it overlaps any damage box at IoU >= 0.5, whatever its type. The pipeline detector is unchanged.'
    (out/'summary.json').write_text(json.dumps(dict(split=f'CarDD {split}', images=n_images, types=TYPES, note=note, results=results), indent=2))

    print(f'\nCLASS-AGNOSTIC, CarDD {split} ({n_images} images)')
    for name, res in results.items():
        b = res['best']
        print(f"{name:20} mAP50 {res['mAP50']:.3f} mAP50-95 {res['mAP50_95']:.3f} | conf {b['confidence']:.2f} "
              f"P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f} FP {b['fp']} no-det {b['no_detection_images']}")
    print('\nRECALL BY DAMAGE TYPE (at each model\'s threshold)')
    print(' ' * 20 + ''.join(f'{k[:13]:>14}' for k in TYPES))
    for name, res in results.items():
        print(f'{name:20}' + ''.join(f"{res['best'][KEY(k)]:14.3f}" for k in TYPES))
    if 'per_type' in results.get(SIX, {}):
        print(f'\nSIX-CLASS PER-TYPE AP50 (mean over types {results[SIX]["mean_over_types_mAP50"]:.3f}, not comparable with class-agnostic mAP50)')
        print(json.dumps(results[SIX]['per_type']))
    print('Saved to', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
