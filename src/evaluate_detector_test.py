"""Single evaluation of the chosen detector on the CarDD test split. Nothing is selected here; run it once."""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from select_detector import DATA, ROOT, RUNS, sha256, true_positives


def ground_truth(image, w, h):
    lab = DATA/'labels/test'/(image.stem + '.txt')
    rows = [list(map(float, l.split()[1:])) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
    return np.array([[(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h] for x, y, bw, bh in rows]).reshape(-1, 4)


def main():
    exp = ROOT/'outputs/experiments'
    if sorted(exp.glob('detector_test_*')):
        sys.exit('A detector test evaluation already exists. The CarDD test split is evaluated only once.')
    sel_dir = sorted(exp.glob('detector_selection_*'))[-1]
    sel = json.loads((sel_dir/'selection.json').read_text())
    chosen = sel['chosen']['model']
    models = {chosen: (ROOT/sel['checkpoint'], sel['chosen']['confidence'], 'chosen'),
              'yolo11n_e10': (RUNS/'yolo11n_initial/weights/best.pt', sel['best_per_model']['yolo11n_e10']['confidence'], 'reference')}
    out = exp/('detector_test_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)
    images = sorted(p for p in (DATA/'images/test').iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    results = {}
    for name, (ckpt, conf, role) in models.items():
        model = YOLO(str(ckpt))
        m = model.val(data=str(DATA/'data.yaml'), split='test', imgsz=640, batch=16, device='mps', conf=0.001, iou=0.7,
                      plots=True, project=str(out), name=name, verbose=False)
        tp = fp = fn = empty = n = 0
        by_count = defaultdict(lambda: [0, 0])
        for r in model.predict(str(DATA/'images/test'), conf=conf, iou=0.7, imgsz=640, batch=16, device='mps', stream=True, verbose=False):
            h, w = r.orig_shape
            boxes, scores, gt = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), ground_truth(Path(r.path), w, h)
            k = true_positives(boxes, scores, gt)
            tp += k; fp += len(boxes) - k; fn += len(gt) - k; empty += int(not len(boxes)); n += 1
            by_count[min(len(gt), 5)][0] += k; by_count[min(len(gt), 5)][1] += len(gt)
        del model
        torch.mps.empty_cache()
        assert n == len(images), f'{name}: predicted {n} of {len(images)} images'
        p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        results[name] = dict(role=role, checkpoint=str(ckpt.relative_to(ROOT)), sha256=sha256(ckpt), confidence=conf,
                             mAP50=round(float(m.box.map50), 4), mAP50_95=round(float(m.box.map), 4),
                             precision=round(p, 4), recall=round(rc, 4), f1=round(2*p*rc / max(p + rc, 1e-9), 4),
                             tp=tp, fp=fp, fn=fn, no_detection_images=empty,
                             recall_by_damages={('5+' if k == 5 else str(k)): round(v[0]/v[1], 4) for k, v in sorted(by_count.items()) if v[1]})
    (out/'summary.json').write_text(json.dumps(dict(split='CarDD test', images=len(images), selection=str(sel_dir.relative_to(ROOT)),
                                                     note='Evaluated once. Model and thresholds were fixed on validation beforehand.',
                                                     results=results), indent=2))
    for name, r in results.items():
        print(f"{r['role']:9} {name:13} conf {r['confidence']:.2f} | mAP50 {r['mAP50']:.3f} mAP50-95 {r['mAP50_95']:.3f} | "
              f"P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} FP {r['fp']} no-det {r['no_detection_images']} | "
              f"recall by damages {r['recall_by_damages']}")
    print('Saved to', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
