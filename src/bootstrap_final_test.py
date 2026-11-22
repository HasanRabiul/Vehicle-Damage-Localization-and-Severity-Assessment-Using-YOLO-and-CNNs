"""Paired bootstrap for the final test comparison: delta_F1 and delta_R_moderate (localized minus whole).
Re-scores the six frozen classifiers of the single test evaluation, checks every per-seed score matches it exactly,
and saves per-image predictions next to it. Nothing is chosen here."""
import csv
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from evaluate_severity import ROOT, SEEDS, VARIANTS
from train_severity_classifier import LABELS, MEAN, STD, Images, metrics

MODERATE = LABELS.index('02-moderate')


def main():
    cfg = json.loads((ROOT/'final_configs.json').read_text())
    crops_dir = ROOT/cfg['localized']['crops']
    test_dir = sorted((ROOT/'outputs/experiments').glob('severity_eval_test_*'))[-1]
    test_cfg = json.loads((test_dir/'config.json').read_text())
    assert test_cfg['crops'] == str((crops_dir/'crops.csv').relative_to(ROOT)), 'latest test evaluation did not use the frozen crops'
    audit = sorted((ROOT/'outputs/experiments').glob('severity_data3a_audit_*'))[-1]
    crops = {r['path']: r for r in csv.DictReader((crops_dir/'crops.csv').open())}
    test = [r for r in csv.DictReader((audit/'manifest.csv').open()) if r['include'] == 'True' and r['split'] == 'test']
    fallback = [crops[r['path']]['fallback'] == 'True' for r in test]
    inputs = {v: [dict(path=r['path'], label=r['label'], fallback=f,
                       image=str(ROOT/r['path'] if v == 'whole' else crops_dir/crops[r['path']]['crop'])) for r, f in zip(test, fallback)]
              for v in VARIANTS}
    saved = {(r['trained_on'], int(r['seed'])): float(r['macro_f1']) for r in csv.DictReader((test_dir/'per_run.csv').open())
             if r['trained_on'] == r['evaluated_on'] and r['subset'] == 'all'}

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    y = np.array([LABELS.index(r['label']) for r in test])
    preds = {}
    for v in VARIANTS:
        for s in SEEDS:
            run = ROOT/test_cfg['runs'][f'{v}_seed{s}']
            model = models.resnet50(weights=None); model.fc = nn.Linear(model.fc.in_features, len(LABELS))
            model.load_state_dict(torch.load(run/'best.pt', map_location='cpu')['state_dict']); model.to(device).eval()
            probs = []
            with torch.no_grad():
                for x, _ in DataLoader(Images(inputs[v], tf), batch_size=64, shuffle=False):
                    probs += model(x.to(device)).softmax(1).cpu().tolist()
            preds[(v, s)] = np.array(probs).argmax(1)
            f1 = metrics(y, preds[(v, s)])['macro_f1']
            assert abs(f1 - saved[(v, s)]) < 1e-9, f'{v} seed {s}: re-scored {f1} differs from saved {saved[(v, s)]}'
            print(f'{v} seed {s}: macro-F1 {f1:.4f} matches the saved test run')

    def deltas(idx):
        f1 = {v: np.mean([metrics(y[idx], preds[(v, s)][idx])['macro_f1'] for s in SEEDS]) for v in VARIANTS}
        mod = y[idx] == MODERATE
        rm = {v: np.mean([(preds[(v, s)][idx][mod] == MODERATE).mean() for s in SEEDS]) for v in VARIANTS}
        return f1['crop'] - f1['whole'], rm['crop'] - rm['whole']

    point = deltas(np.arange(len(y)))
    rng = np.random.default_rng(812)  # same seed and draw order as evaluate_severity.py, so the resamples are identical
    draws = np.array([deltas(rng.integers(0, len(y), len(y))) for _ in range(2000)])
    result = {}
    for i, name in enumerate(['delta_F1', 'delta_R_moderate']):
        result[name] = dict(estimate=round(float(point[i]), 4), ci95_low=round(float(np.percentile(draws[:, i], 2.5)), 4),
                            ci95_high=round(float(np.percentile(draws[:, i], 97.5)), 4),
                            fraction_draws_above_zero=round(float((draws[:, i] > 0).mean()), 4))

    with (test_dir/'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'label', 'fallback'] + [f'{v}_seed{s}' for v in VARIANTS for s in SEEDS])
        for i, r in enumerate(test):
            w.writerow([r['path'], r['label'], fallback[i]] + [LABELS[preds[(v, s)][i]] for v in VARIANTS for s in SEEDS])
    (test_dir/'final_comparison.json').write_text(json.dumps(dict(
        definition='localized minus whole; each side is the mean over seeds 812-814; paired percentile bootstrap over the 245 test images, 2000 resamples, seed 812',
        final_configs='final_configs.json', **result), indent=2))
    print(json.dumps(result, indent=2)); print('Saved predictions.csv and final_comparison.json in', test_dir.relative_to(ROOT))


if __name__ == '__main__':
    main()
