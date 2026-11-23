"""CarDD with the six damage types kept as classes, for the six-class detector comparison. Images are shared with
data/processed/cardd_yolo through a symlink; only the class ids in the labels differ."""
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/cardd/CarDD/CarDD_COCO'
MERGED = ROOT/'data/processed/cardd_yolo'
OUTPUT = ROOT/'data/processed/cardd_yolo6'
SPLITS = {'train2017': 'train', 'val2017': 'val', 'test2017': 'test'}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT/'images').exists():
        shutil.copytree(MERGED/'images', OUTPUT/'images')  # a real copy: YOLO resolves symlinks and would read the one-class labels
    names = None
    for src, split in SPLITS.items():
        coco = json.loads((SOURCE/'annotations'/f'instances_{src}.json').read_text())
        cats = sorted(coco['categories'], key=lambda c: c['id'])
        assert names in (None, [c['name'] for c in cats])
        names, ids = [c['name'] for c in cats], {c['id']: k for k, c in enumerate(cats)}
        by_image = defaultdict(list)
        for a in coco['annotations']:
            by_image[a['image_id']].append(a)
        out = OUTPUT/'labels'/split
        out.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        for im in coco['images']:
            W, H, lines = im['width'], im['height'], []
            for a in by_image[im['id']]:
                x, y, w, h = map(float, a['bbox'])
                x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
                    continue
                k = ids[a['category_id']]
                counts[names[k]] += 1
                lines.append(f'{k} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
            (out/(Path(im['file_name']).stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        merged = sum(len(p.read_text().split()) // 5 for p in (MERGED/'labels'/split).glob('*.txt'))
        assert sum(counts.values()) == merged, (split, sum(counts.values()), merged)
        print(split, len(coco['images']), 'images', dict(counts))
    with (OUTPUT/'data.yaml').open('w') as f:
        yaml.safe_dump(dict(path=str(OUTPUT), train='images/train', val='images/val', test='images/test',
                            nc=len(names), names=dict(enumerate(names))), f, sort_keys=False)
    print('Saved', OUTPUT/'data.yaml', names)


if __name__ == '__main__':
    main()
