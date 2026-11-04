"""Reproduce initial checkpoint validation without training or using test data."""
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'runs/detect/runs/damage_detection/yolo11n_initial/weights/best.pt'
DATA = ROOT / 'data/processed/cardd_yolo/data.yaml'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'outputs/experiments' / ('yolo11n_baseline_val_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    config = dict(data=str(DATA), split='val', imgsz=640, batch=8, device=device,
                  workers=0, conf=0.001, iou=0.7, max_det=300, plots=True,
                  save_json=True, save_txt=False, augment=False, half=False,
                  seed=0, deterministic=True, project=str(out), name='validation', exist_ok=False)
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, executable=sys.executable, platform=platform.platform(),
                    packages={p: importlib.metadata.version(p) for p in
                              ['torch', 'torchvision', 'ultralytics', 'numpy', 'pillow']},
                    mps_available=torch.backends.mps.is_available(), device=device,
                    checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=sha256(CHECKPOINT),
                    dataset_yaml_sha256=sha256(DATA), script_sha256=sha256(Path(__file__)),
                    note='Validation only. No training. Test split not evaluated. CPU/MPS may differ slightly.')
    (out / 'environment.json').write_text(json.dumps(metadata, indent=2))
    (out / 'validation_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (out / 'dataset_config.yaml').write_text(DATA.read_text())
    print('Experiment:', out, flush=True)
    print('Device:', device, flush=True)
    try:
        metrics = YOLO(str(CHECKPOINT)).val(**config)
        result = dict(metrics.results_dict)
        # Optional external COCO evaluation may emit zero placeholders when unavailable.
        for size in ['small', 'medium', 'large']:
            result.pop(f'metrics/mAP_{size}(B)', None)
        result['evaluation_backend'] = 'Ultralytics internal evaluator; external COCO size metrics excluded'
        result['speed_ms_per_image'] = metrics.speed
        result['images'] = len(list((ROOT / 'data/processed/cardd_yolo/images/val').glob('*.jpg')))
        result['confusion_matrix'] = metrics.confusion_matrix.matrix.tolist()
        result['confusion_matrix_axes'] = 'Rows predicted, columns ground truth; class order damage, background'
        result['confusion_matrix_confidence'] = 0.001
        result['confusion_matrix_matching_iou'] = 0.45
        np.savetxt(out / 'confusion_matrix.csv', metrics.confusion_matrix.matrix, delimiter=',', fmt='%.0f',
                   header='true_damage,true_background', comments='')
        (out / 'metrics.json').write_text(json.dumps(result, indent=2))
        with (ROOT / 'runs/detect/runs/damage_detection/yolo11n_initial/results.csv').open() as f:
            prior = list(csv.DictReader(f))[-1]
        keys = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        comparison = {k: {'saved_epoch_10':float(prior[k]), 'reproduced':result[k],
                          'difference':result[k]-float(prior[k])} for k in keys}
        (out / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        metadata.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                        checkpoint_unchanged=sha256(CHECKPOINT)==metadata['checkpoint_sha256'])
        print(json.dumps(comparison, indent=2), flush=True)
    except Exception as exc:
        metadata.update(status='failed', error=repr(exc))
        raise
    finally:
        (out / 'environment.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
