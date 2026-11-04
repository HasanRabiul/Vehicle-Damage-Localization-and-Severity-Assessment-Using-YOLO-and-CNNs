"""Local, read-only research dashboard. Run: python3 webapp/app.py"""
from pathlib import Path
from io import BytesIO
import base64
import csv
import json
import threading

from flask import Flask, jsonify, render_template, request, send_file, abort
from PIL import Image, UnidentifiedImageError

from catalog import figure_groups, gallery_groups, gallery_extensions
from pipeline import pipeline

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/detect/runs/damage_detection/yolo11n_initial'
CHECKPOINT = RUN / 'weights/best.pt'
SEVERITY_CHECKPOINT = ROOT / 'models/resnet50/severity_resnet50_whole_seed813.pt'
EXPERIMENTS = ROOT / 'outputs/experiments'
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
lock = threading.Lock()
model = None
severity_model = None

def severity_results():
    """Planned whole- vs crop-input comparison from the latest saved evaluations; None when not yet run."""
    out = {}
    for split in ['validation', 'test']:
        runs = sorted(EXPERIMENTS.glob(f'severity_eval_{split}_*'))
        if not runs or not (runs[-1] / 'summary.json').exists():
            continue
        s = json.loads((runs[-1] / 'summary.json').read_text())
        ci = s.get('bootstrap', {}).get('intervals', {})
        out[split] = dict(images=s['images'], run=str(runs[-1].relative_to(ROOT)), majority_accuracy=s['majority_baseline_accuracy'],
                          difference_ci=ci.get('difference_whole_minus_crop/macro_f1'),
                          variants={v: dict(macro_f1=s['results'][f'{v}->{v}']['all']['macro_f1']['mean'],
                                            macro_f1_std=s['results'][f'{v}->{v}']['all']['macro_f1']['std'],
                                            accuracy=s['results'][f'{v}->{v}']['all']['accuracy']['mean'],
                                            ci=ci.get(f'{v}->{v}/macro_f1')) for v in ['whole', 'crop']})
    return out or None

def classify_severity(im):
    global severity_model
    import torch
    from torch import nn
    from torchvision import models, transforms
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if severity_model is None:
        state = torch.load(SEVERITY_CHECKPOINT, map_location='cpu')
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(state['labels']))
        net.load_state_dict(state['state_dict'])
        severity_model = (net.to(device).eval(), state)
    net, state = severity_model
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize(state['mean'], state['std'])])
    with torch.no_grad():
        probs = net(tf(im).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
    labels = [l.split('-', 1)[1] for l in state['labels']]
    best = max(range(len(probs)), key=probs.__getitem__)
    return dict(label=labels[best], confidence=probs[best], probabilities=dict(zip(labels, probs)),
                input='whole image', checkpoint=str(SEVERITY_CHECKPOINT.relative_to(ROOT)))

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
    return jsonify(history=rows, splits=splits, checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                   checkpoint_exists=CHECKPOINT.exists(),
                   severity_checkpoint=str(SEVERITY_CHECKPOINT.relative_to(ROOT)),
                   severity_checkpoint_exists=SEVERITY_CHECKPOINT.exists(),
                   severity=severity_results(),
                   figure_groups=figure_groups(ROOT),
                   galleries=gallery_groups(ROOT),
                   pipeline=pipeline(ROOT),
                   downloads=[p for p in files if p.endswith(('.csv','.yaml','.json','.txt'))],
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
    global model
    upload = request.files.get('image')
    if not upload:
        return jsonify(error='Choose an image first.'), 400
    try:
        conf = float(request.form.get('confidence', '.25'))
        iou = float(request.form.get('iou', '.7'))
        if not 0.001 <= conf <= 1 or not .1 <= iou <= .95:
            raise ValueError('Thresholds are outside the supported range.')
        im = Image.open(upload.stream)
        if im.width * im.height > 25_000_000:
            raise ValueError('Image exceeds 25 megapixels. Please resize it.')
        im = im.convert('RGB')
        from PIL import ImageOps
        # Normalize orientation before inference and rendering.
        upload.stream.seek(0)
        im = ImageOps.exif_transpose(Image.open(upload.stream)).convert('RGB')
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return jsonify(error=str(exc)), 400
    if not CHECKPOINT.exists():
        return jsonify(error='The configured checkpoint is missing.'), 404
    if not lock.acquire(blocking=False):
        return jsonify(error='A prediction is already running. Please try again shortly.'), 409
    try:
        import torch
        from ultralytics import YOLO
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if model is None:
            model = YOLO(str(CHECKPOINT))
        result = model.predict(im, conf=conf, iou=iou, imgsz=640, device=device, save=False, verbose=False)[0]
        rendered = Image.fromarray(result.plot()[:, :, ::-1])
        buffer = BytesIO()
        rendered.save(buffer, format='JPEG', quality=90)
        boxes = [dict(class_name=result.names[int(b.cls.item())], confidence=float(b.conf.item()),
                      xyxy=[round(v, 2) for v in b.xyxy[0].tolist()]) for b in result.boxes]
        # Severity uses the whole image: on validation and test it outperformed YOLO-crop input.
        severity = classify_severity(im) if SEVERITY_CHECKPOINT.exists() else None
        return jsonify(image='data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                       boxes=boxes, device=device, speed=result.speed, confidence=conf, iou=iou,
                       checkpoint=str(CHECKPOINT.relative_to(ROOT)), severity=severity)
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
