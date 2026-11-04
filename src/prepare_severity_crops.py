"""Create YOLO damage crops for the data3a severity manifest; no metrics are computed and originals are only read."""
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

CHECKPOINT=ROOT/'runs/detect/runs/damage_detection/yolo11n_initial/weights/best.pt'
CONF=.25
PAD=.10
SPLITS=['training','validation','test']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit=sorted((ROOT/'outputs/experiments').glob('severity_data3a_audit_*'))[-1]
    manifest=[r for r in csv.DictReader((audit/'manifest.csv').open()) if r['include']=='True']
    out=ROOT/'outputs/experiments'/('severity_crops_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint_hash=sha256(CHECKPOINT)
    model=YOLO(str(CHECKPOINT))
    records=[]
    for r in manifest:
        src=ROOT/r['path']
        with Image.open(src) as image:im=image.convert('RGB')
        W,H=im.size
        res=model.predict(im,imgsz=640,conf=CONF,iou=.7,max_det=300,device=device,verbose=False)[0]
        boxes=res.boxes.xyxy.cpu().numpy();scores=res.boxes.conf.cpu().numpy()
        if len(boxes):
            # Union of all detections, padded by PAD of its size on each side and clipped to the image.
            x1,y1=boxes[:,:2].min(0);x2,y2=boxes[:,2:].max(0)
            px,py=PAD*(x2-x1),PAD*(y2-y1)
            box=[max(0,int(np.floor(x1-px))),max(0,int(np.floor(y1-py))),min(W,int(np.ceil(x2+px))),min(H,int(np.ceil(y2+py)))]
            fallback=False
        else:
            box=[0,0,W,H];fallback=True
        rel=Path('crops')/r['split']/r['label']/(src.stem+'_'+src.suffix[1:]+'.png')
        (out/rel).parent.mkdir(parents=True,exist_ok=True)
        im.crop(box).save(out/rel)
        records.append(dict(path=r['path'],split=r['split'],label=r['label'],crop=str(rel),fallback=fallback,detections=len(boxes),
                            max_score=round(float(scores.max()),4) if len(boxes) else '',crop_x1=box[0],crop_y1=box[1],crop_x2=box[2],crop_y2=box[3],
                            image_width=W,image_height=H,crop_width=box[2]-box[0],crop_height=box[3]-box[1],
                            crop_area_fraction=round((box[2]-box[0])*(box[3]-box[1])/(W*H),4),
                            boxes=json.dumps([[round(float(v),1) for v in b] for b in boxes])))
    with (out/'crops.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    summary={}
    for s in SPLITS:
        rs=[x for x in records if x['split']==s]
        summary[s]=dict(images=len(rs),fallback=sum(x['fallback'] for x in rs),fallback_rate=round(sum(x['fallback'] for x in rs)/len(rs),4),
                        fallback_by_label=dict(sorted(Counter(x['label'] for x in rs if x['fallback']).items())),
                        median_crop_area_fraction=float(np.median([x['crop_area_fraction'] for x in rs if not x['fallback']])),
                        median_crop_short_side=float(np.median([min(x['crop_width'],x['crop_height']) for x in rs if not x['fallback']])),
                        crops_short_side_under_64=sum(min(x['crop_width'],x['crop_height'])<64 for x in rs if not x['fallback']),
                        detections_per_image=dict(sorted(Counter(x['detections'] for x in rs).items())))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    # Sample panel from training/validation only: four detected and two fallback images per class.
    rng=random.Random(812);picks=[]
    for label in sorted({x['label'] for x in records}):
        pool=[x for x in records if x['label']==label and x['split']!='test']
        det=[x for x in pool if not x['fallback']];fb=[x for x in pool if x['fallback']]
        picks+=rng.sample(det,min(4,len(det)))+rng.sample(fb,min(2,len(fb)))
    fig,axes=plt.subplots(-(-len(picks)//3),6,figsize=(18,2.6*-(-len(picks)//3)+.8),squeeze=False,constrained_layout=True)
    for ax in axes.flat:ax.axis('off')
    for n,x in enumerate(picks):
        a,b=axes[n//3,2*(n%3)],axes[n//3,2*(n%3)+1]
        with Image.open(ROOT/x['path']) as image:a.imshow(image.convert('RGB'))
        for bx in json.loads(x['boxes']):a.add_patch(Rectangle(bx[:2],bx[2]-bx[0],bx[3]-bx[1],fill=False,edgecolor='#e68621',linewidth=1.5))
        a.add_patch(Rectangle((x['crop_x1'],x['crop_y1']),x['crop_width'],x['crop_height'],fill=False,edgecolor='#2877d5',linewidth=2,linestyle='--'))
        a.set_title(f"{x['split']}/{x['label']}/{Path(x['path']).name}",fontsize=7)
        with Image.open(out/x['crop']) as image:b.imshow(image.convert('RGB'))
        b.set_title(('FALLBACK whole image' if x['fallback'] else f"crop {x['crop_width']}x{x['crop_height']} · {x['detections']} det")+'',fontsize=7)
    fig.suptitle(f'YOLO crops for severity (training/validation samples) · orange: detections conf > {CONF} · blue dashed: padded union crop',fontsize=11)
    fig.savefig(out/'crop_samples.png',dpi=130);plt.close(fig)
    (out/'config.json').write_text(json.dumps(dict(manifest=str((audit/'manifest.csv').relative_to(ROOT)),manifest_sha256=sha256(audit/'manifest.csv'),
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=checkpoint_hash,checkpoint_unchanged=sha256(CHECKPOINT)==checkpoint_hash,
        imgsz=640,conf=CONF,iou=.7,max_det=300,device=device,crop='union of detections, padded 10% per side, clipped',fallback='whole image when no detection',
        crop_format='PNG (lossless)',panel_seed=812,script_sha256=sha256(Path(__file__)),python=sys.version,platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','ultralytics','numpy','pillow']}),indent=2))
    (out/'README.md').write_text(f'# YOLO crops for severity classification\n\nThe yolo11n_initial checkpoint was run on every included image of the data3a manifest (imgsz 640, confidence > {CONF}, NMS IoU 0.7). Each image produces one crop: the union of its detections, padded by 10% of the union size per side and clipped. Images with no detection fall back to the whole image and are flagged in crops.csv; they are kept, not discarded.\n\nCrops for the test split are created so the pipeline can be evaluated later; no metrics were computed on any split. crop_samples.png shows training/validation examples only. Original images and the checkpoint are unchanged.\n')
    print('OUTPUT',out);print(json.dumps(summary,indent=1))

if __name__=='__main__':main()
