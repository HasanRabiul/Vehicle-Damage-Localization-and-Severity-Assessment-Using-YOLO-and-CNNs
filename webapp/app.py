"""Local, read-only research dashboard. Run: python3 webapp/app.py"""
from pathlib import Path
from io import BytesIO
import base64
import csv
import json
import math
import threading

from flask import Flask, jsonify, render_template, request, send_file, abort
from PIL import Image, ImageOps, UnidentifiedImageError

from catalog import figure_groups, gallery_groups, gallery_extensions
from findings import dashboard_models, findings
from pipeline import pipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS = dashboard_models(ROOT)
CHECKPOINT = ROOT / MODELS['detector']             # frozen pipeline detector (final_configs.json)
RUN = CHECKPOINT.parents[1]
TYPES_CHECKPOINT = ROOT / MODELS['types_detector']  # six-class study model, display only
SEVERITY = {v: ROOT / MODELS[v]['checkpoint'] for v in ['whole', 'crop'] if MODELS.get(v)}
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
lock = threading.Lock()
detectors, classifiers = {}, {}

def to_data_url(im, quality=90):
    buffer = BytesIO()
    im.save(buffer, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()

def detector(path):
    from ultralytics import YOLO
    if path not in detectors:
        detectors[path] = YOLO(str(path))
    return detectors[path]

def classify_severity(im, variant, device):
    import torch
    from torch import nn
    from torchvision import models, transforms
    if variant not in classifiers:
        state = torch.load(SEVERITY[variant], map_location='cpu')
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(state['labels']))
        net.load_state_dict(state['state_dict'])
        classifiers[variant] = (net.to(device).eval(), state)
    net, state = classifiers[variant]
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize(state['mean'], state['std'])])
    with torch.no_grad():
        probs = net(tf(im).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
    labels = [l.split('-', 1)[1] for l in state['labels']]
    best = max(range(len(probs)), key=probs.__getitem__)
    return dict(label=labels[best], confidence=probs[best], probabilities=dict(zip(labels, probs)),
                seed=MODELS[variant]['seed'], checkpoint=MODELS[variant]['checkpoint'])

def union_crop(im, boxes):
    """Same rule as src/prepare_severity_crops.py: union of detections padded per side, clipped; whole image if none."""
    W, H = im.size
    if not boxes:
        return [0, 0, W, H], True
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    px, py = MODELS['pad'] * (x2 - x1), MODELS['pad'] * (y2 - y1)
    return [max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py)), min(W, math.ceil(x2 + px)), min(H, math.ceil(y2 + py))], False

def artifacts():
    roots = [ROOT / 'runs', ROOT / 'outputs']
    return {str(p.relative_to(ROOT)): p for root in roots for p in root.rglob('*')
            if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.csv', '.yaml', '.json', '.txt'}}

@app.get('/')
def index():
    return render_template('index.html')

@app.get('/api/overview')
def overview():
    rows = []
    if (RUN / 'results.csv').exists():
        with (RUN / 'results.csv').open() as f:
            rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    splits = []
    for split in ['train', 'val', 'test']:
        root = ROOT / 'data/processed/cardd_yolo'
        images = list((root / 'images' / split).glob('*'))
        labels = list((root / 'labels' / split).glob('*.txt'))
        splits.append(dict(split=split, images=sum(p.suffix.lower() in {'.jpg','.jpeg','.png'} for p in images),
                           boxes=sum(len(p.read_text().splitlines()) for p in labels)))
    files = artifacts()
    checkpoints = [dict(role=role, path=str(p.relative_to(ROOT)), exists=p.exists()) for role, p in
                   [('Detector (frozen pipeline)', CHECKPOINT), ('Detector (six damage types, display only)', TYPES_CHECKPOINT)]
                   + [(f'Severity · {v} input · seed {MODELS[v]["seed"]}', SEVERITY[v]) for v in SEVERITY]]
    return jsonify(history=rows, splits=splits, run=RUN.name, checkpoints=checkpoints, models=MODELS,
                   findings=findings(ROOT),
                   figure_groups=figure_groups(ROOT),
                   galleries=gallery_groups(ROOT),
                   pipeline=pipeline(ROOT),
                   downloads=[p for p in files if p.endswith(('.csv','.yaml','.json','.txt'))],
                   final_configs=(ROOT / 'final_configs.json').read_text() if (ROOT / 'final_configs.json').exists() else '',
                   config=(RUN / 'args.yaml').read_text() if (RUN / 'args.yaml').exists() else '')

@app.get('/api/gallery')
def gallery():
    folder = request.args.get('folder', '')
    query = request.args.get('q', '').lower()
    ext = gallery_extensions(ROOT, folder)
    if ext is None:
        return jsonify(total=0, files=[])
    files = sorted(p for p in artifacts() if str(Path(p).parent) == folder and p.lower().endswith(ext) and query in Path(p).name.lower())
    page = max(0, request.args.get('page', 0, type=int))
    return jsonify(total=len(files), files=files[page*24:(page+1)*24])

@app.get('/artifact/<path:name>')
def artifact(name):
    p = artifacts().get(name)
    if p is None or p.is_symlink() or not p.resolve().is_relative_to(ROOT):
        abort(404)
    return send_file(p)

@app.post('/api/predict')
def predict():
    upload = request.files.get('image')
    if not upload:
        return jsonify(error='Choose an image first.'), 400
    try:
        conf = float(request.form.get('confidence', MODELS['confidence']))
        iou = float(request.form.get('iou', '.7'))
        mode = request.form.get('detector', 'pipeline')
        if not 0.001 <= conf <= 1 or not .1 <= iou <= .95 or mode not in {'pipeline', 'types'}:
            raise ValueError('Settings are outside the supported range.')
        im = Image.open(upload.stream)
        if im.width * im.height > 25_000_000:
            raise ValueError('Image exceeds 25 megapixels. Please resize it.')
        # Normalize orientation before inference and rendering.
        upload.stream.seek(0)
        im = ImageOps.exif_transpose(Image.open(upload.stream)).convert('RGB')
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return jsonify(error=str(exc)), 400
    shown = CHECKPOINT if mode == 'pipeline' else TYPES_CHECKPOINT
    if not CHECKPOINT.exists() or not shown.exists():
        return jsonify(error='A configured detector checkpoint is missing.'), 404
    if not lock.acquire(blocking=False):
        return jsonify(error='A prediction is already running. Please try again shortly.'), 409
    try:
        import torch
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        # The localized severity input always uses the frozen detector and threshold, whatever the display settings.
        frozen = detector(CHECKPOINT).predict(im, conf=MODELS['confidence'], iou=.7, imgsz=640, max_det=300, device=device, verbose=False)[0]
        crop_box, fallback = union_crop(im, frozen.boxes.xyxy.cpu().tolist())
        result = detector(shown).predict(im, conf=conf, iou=iou, imgsz=640, device=device, verbose=False)[0]
        rendered = to_data_url(Image.fromarray(result.plot()[:, :, ::-1]))
        boxes = [dict(class_name=result.names[int(b.cls.item())], confidence=float(b.conf.item()),
                      xyxy=[round(v, 2) for v in b.xyxy[0].tolist()]) for b in result.boxes]
        crop = im.crop(crop_box)
        thumb = crop.copy()
        thumb.thumbnail((640, 640))
        severity = {v: classify_severity(crop if v == 'crop' else im, v, device) for v in SEVERITY}
        return jsonify(image=rendered, boxes=boxes, device=device, speed=result.speed, confidence=conf, iou=iou, mode=mode,
                       checkpoint=str(shown.relative_to(ROOT)), severity=severity,
                       localized=dict(image=to_data_url(thumb, 85), box=crop_box, fallback=fallback, detections=len(frozen.boxes),
                                      confidence=MODELS['confidence'], pad=MODELS['pad'],
                                      area_fraction=(crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]) / (im.width * im.height)))
    except Exception as exc:
        app.logger.exception('Inference failed')
        return jsonify(error=f'Inference failed: {exc}'), 500
    finally:
        lock.release()

@app.errorhandler(413)
def too_large(exc):
    return jsonify(error='Upload exceeds the 16 MB limit.'), 413

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8120, debug=False)
