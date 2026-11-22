"""Error analysis of the final test comparison; nothing is chosen here. Per-class metrics, detected vs fallback and crop
size, Grad-CAM attention inside the YOLO damage boxes (whole-image models), and moderate-class errors."""
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from evaluate_severity import SEEDS, VARIANTS
from train_severity_classifier import LABELS, MEAN, STD, metrics

SHORT = [l.split('-', 1)[1] for l in LABELS]
VIEW = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class GradCAM:
    """Grad-CAM on ResNet50 layer4 for the predicted class."""
    def __init__(self, model):
        self.model, self.acts = model, None
        model.layer4.register_forward_hook(lambda m, i, o: setattr(self, 'acts', o))

    def __call__(self, x):
        out = self.model(x)
        cls = int(out.argmax(1))
        self.acts.retain_grad()
        out[0, cls].backward()
        a, g = self.acts[0], self.acts.grad[0]
        cam = torch.relu((g.mean((1, 2))[:, None, None] * a).sum(0))
        cam = nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), cls


def box_mask(boxes, size):
    """Union of detection boxes mapped into the 224x224 Resize(256)+CenterCrop(224) view."""
    w, h = size; k = 256 / min(w, h); W, H = int(w * k), int(h * k)
    ox, oy = int(round((W - 224) / 2)), int(round((H - 224) / 2))
    m = np.zeros((224, 224))
    for x1, y1, x2, y2 in boxes:
        a, b = max(0, int(x1 * k - ox)), max(0, int(y1 * k - oy))
        c, d = min(224, int(np.ceil(x2 * k - ox))), min(224, int(np.ceil(y2 * k - oy)))
        if c > a and d > b:
            m[b:d, a:c] = 1
    return m


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    cfg = json.loads((ROOT/'final_configs.json').read_text())
    crops_dir = ROOT/cfg['localized']['crops']
    test_dir = sorted((ROOT/'outputs/experiments').glob('severity_eval_test_*'))[-1]
    test_cfg = json.loads((test_dir/'config.json').read_text())
    rows = list(csv.DictReader((test_dir/'predictions.csv').open()))
    crops = {r['path']: r for r in csv.DictReader((crops_dir/'crops.csv').open())}
    assert len(rows) == 245, len(rows)
    out = ROOT/'outputs/experiments'/('severity_error_analysis_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True)

    y = np.array([LABELS.index(r['label']) for r in rows])
    P = {(v, s): np.array([LABELS.index(r[f'{v}_seed{s}']) for r in rows]) for v in VARIANTS for s in SEEDS}
    fb = np.array([r['fallback'] == 'True' for r in rows])
    area = np.array([float(crops[r['path']]['crop_area_fraction']) for r in rows])
    ndet = np.array([int(crops[r['path']]['detections']) for r in rows])
    vote = lambda v, i: Counter(int(P[(v, s)][i]) for s in SEEDS).most_common(1)[0][0]

    # 1. Per-class metrics, mean over seeds.
    per_class = []
    for v in VARIANTS:
        for k, l in enumerate(LABELS):
            vals = {m: np.mean([metrics(y, P[(v, s)])['per_class'][l][m] for s in SEEDS]) for m in ['precision', 'recall', 'f1']}
            per_class.append(dict(input=v, label=SHORT[k], support=int((y == k).sum()), **{m: round(float(x), 4) for m, x in vals.items()}))
    write_csv(out/'per_class.csv', per_class)

    # 2. Where does localization help or hurt?
    subsets = {'all': np.ones(len(y), bool), 'detected': ~fb, 'fallback': fb,
               'detected, 1 box': ~fb & (ndet == 1), 'detected, 2+ boxes': ~fb & (ndet >= 2),
               'crop < 1/3 of image': ~fb & (area < 1/3), 'crop 1/3 to 2/3': ~fb & (area >= 1/3) & (area < 2/3),
               'crop >= 2/3 of image': ~fb & (area >= 2/3)}
    sub_rows = []
    for name, m in subsets.items():
        r = dict(subset=name, n=int(m.sum()))
        for v in VARIANTS:
            r[f'{v}_macro_f1'] = round(float(np.mean([metrics(y[m], P[(v, s)][m])['macro_f1'] for s in SEEDS])), 4)
            r[f'{v}_accuracy'] = round(float(np.mean([(y[m] == P[(v, s)][m]).mean() for s in SEEDS])), 4)
        r['crop_minus_whole_f1'] = round(r['crop_macro_f1'] - r['whole_macro_f1'], 4)
        sub_rows.append(r)
    write_csv(out/'subsets.csv', sub_rows)

    # 3. Where do moderate images go?
    mod = y == LABELS.index('02-moderate')
    moderate_to = {v: {SHORT[c]: round(float(np.mean([(P[(v, s)][mod] == c).mean() for s in SEEDS])), 4) for c in range(3)} for v in VARIANTS}

    # 4. Grad-CAM of the whole-image models on detected images: attention inside the YOLO boxes relative to box area.
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    detected = [i for i in range(len(rows)) if not fb[i]]
    ratios, cams, cam_rows = defaultdict(list), {}, []
    for s in SEEDS:
        model = models.resnet50(weights=None); model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        model.load_state_dict(torch.load(ROOT/test_cfg['runs'][f'whole_seed{s}']/'best.pt', map_location='cpu')['state_dict'])
        model.to(device).eval()
        gc = GradCAM(model)
        for i in detected:
            with Image.open(ROOT/rows[i]['path']) as im:
                im = im.convert('RGB')
            view = VIEW(im)
            heat, _ = gc(NORM(view)[None].to(device))
            mask = box_mask(json.loads(crops[rows[i]['path']]['boxes']), im.size)
            inside = mask.mean()
            if not 0.02 < inside < 0.95:
                continue  # boxes cover almost none or almost all of the view: the ratio says nothing
            ratio = float((heat * mask).sum() / heat.sum() / inside)
            ratios[s].append(ratio)
            cam_rows.append(dict(path=rows[i]['path'], seed=s, box_area_fraction=round(float(inside), 4), attention_ratio=round(ratio, 4)))
            if s == 813:
                cams[i] = (np.asarray(view), heat, mask, ratio)
        del model, gc
        if device == 'mps':
            torch.mps.empty_cache()
    write_csv(out/'gradcam_attention.csv', cam_rows)
    allr = np.concatenate([ratios[s] for s in SEEDS])
    gradcam = dict(images=len(ratios[813]), mean_ratio=round(float(allr.mean()), 3), median_ratio=round(float(np.median(allr)), 3),
                   fraction_above_1=round(float((allr > 1).mean()), 3),
                   per_seed={s: round(float(np.mean(ratios[s])), 3) for s in SEEDS},
                   meaning='attention inside YOLO boxes divided by box area; 1 = no preference for the damage, >1 = focuses on it')

    # Panels.
    rng = np.random.default_rng(812)
    pick = sorted(rng.choice(sorted(cams), size=min(8, len(cams)), replace=False), key=lambda i: y[i])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.5))
    for ax in axes.flat:
        ax.axis('off')
    for n, i in enumerate(pick):
        view, heat, mask, ratio = cams[i]
        a, b = axes.flat[2*n], axes.flat[2*n + 1]
        a.imshow(view); a.contour(mask, [0.5], colors='lime', linewidths=1.5)
        a.set_title(f'true {SHORT[y[i]]} · pred {SHORT[vote("whole", i)]}', fontsize=9)
        b.imshow(view); b.imshow(heat, cmap='jet', alpha=.45); b.contour(mask, [0.5], colors='white', linewidths=1)
        b.set_title(f'Grad-CAM · ratio {ratio:.2f}', fontsize=9)
    fig.suptitle('Whole-image ResNet50 (seed 813) on test images · green/white: YOLO11s damage boxes · ratio > 1: attention concentrated on the damage', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'gradcam.png', dpi=150); plt.close(fig)

    wrong = [i for i in np.where(mod)[0] if vote('whole', i) != y[i]][:12]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
    for ax in axes.flat:
        ax.axis('off')
    for ax, i in zip(axes.flat, wrong):
        with Image.open(ROOT/rows[i]['path']) as im:
            ax.imshow(im.convert('RGB'))
        for x1, y1, x2, y2 in json.loads(crops[rows[i]['path']]['boxes']):
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='lime', lw=1.5))
        ax.set_title(f'whole → {SHORT[vote("whole", i)]} · crop → {SHORT[vote("crop", i)]}', fontsize=9)
    fig.suptitle('Moderate test images misclassified by the whole-image model (majority of 3 seeds) · green: YOLO11s boxes', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'moderate_errors.png', dpi=150); plt.close(fig)

    summary = dict(test_evaluation=str(test_dir.relative_to(ROOT)), per_class=per_class, subsets=sub_rows,
                   moderate_predicted_as=moderate_to, gradcam=gradcam)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'README.md').write_text('# Error analysis of the final test comparison\n\nAnalysis of the fixed test predictions in '
        f'{test_dir.relative_to(ROOT)}; no model or setting was chosen here. Grad-CAM uses the whole-image models on test images '
        'where YOLO11s found damage; the ratio is the share of attention inside the boxes divided by the boxes\' share of the view.\n')

    print('PER CLASS (mean of 3 seeds)'); [print(r) for r in per_class]
    print('\nSUBSETS'); [print(r) for r in sub_rows]
    print('\nMODERATE PREDICTED AS', json.dumps(moderate_to))
    print('\nGRAD-CAM', json.dumps(gradcam))
    print('Saved to', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
