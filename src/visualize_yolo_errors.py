"""Create deterministic validation error examples from frozen predictions."""
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
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
from ultralytics.utils.metrics import ConfusionMatrix


def main():
    audit=sorted((ROOT/'outputs/experiments').glob('yolo11n_threshold_audit_*'))[-1]
    config=json.loads((audit/'config.json').read_text())
    source=ROOT/config['predictions']
    coco=json.loads((ROOT/config['annotations']).read_text())
    preds=defaultdict(list)
    for p in json.loads(source.read_text()):
        if p['score']>.25:preds[p['image_id']].append(p)
    out=ROOT/'outputs/experiments'/('yolo11n_error_analysis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    records=[]; details={}
    for im in sorted(coco['images'],key=lambda x:x['file_name']):
        name=im['file_name']; ps=preds[im['id']]
        pb=np.array([p['bbox'] for p in ps],dtype=np.float32).reshape(-1,4);pb[:,2:]+=pb[:,:2]
        gb=[]
        for line in (ROOT/'data/processed/cardd_yolo/labels/val'/(Path(name).stem+'.txt')).read_text().splitlines():
            _,xc,yc,w,h=map(float,line.split());gb.append([(xc-w/2)*im['width'],(yc-h/2)*im['height'],(xc+w/2)*im['width'],(yc+h/2)*im['height']])
        gb=np.array(gb,dtype=np.float32).reshape(-1,4)
        cm=ConfusionMatrix(names={0:'damage'},save_matches=True)
        cm.process_batch(dict(cls=torch.zeros(len(ps)),conf=torch.tensor([p['score'] for p in ps]),bboxes=torch.tensor(pb)),dict(cls=torch.zeros(len(gb)),bboxes=torch.tensor(gb)),conf=.25,iou_thres=.45)
        # Repeat the installed validator's greedy matching to retain indices for drawing.
        from ultralytics.utils.metrics import box_iou
        ious=box_iou(torch.tensor(gb),torch.tensor(pb)).numpy()
        matched_gt=set();matched_pred=set()
        for j in np.argsort(-np.array([p['score'] for p in ps]),kind='stable'):
            if len(gb)==0:continue
            k=int(ious[:,j].argmax())
            if ious[k,j]>.45:matched_gt.add(k);matched_pred.add(int(j));ious[k,:]=0
        tp,fp,fn=int(cm.matrix[0,0]),int(cm.matrix[0,1]),int(cm.matrix[1,0])
        assert len(matched_pred)==tp and len(ps)-tp==fp and len(gb)-tp==fn
        row=dict(image=name,tp=tp,fp=fp,fn=fn,detections=len(ps),ground_truth=len(gb))
        records.append(row);details[name]=(ps,pb,gb,matched_pred,matched_gt)
    pools={
        'correct_localization':[r for r in records if r['tp']>0 and r['fp']==0 and r['fn']==0],
        'false_positives':[r for r in records if r['fp']>0],
        'missed_damage':[r for r in records if r['fn']>0 and r['detections']>0],
        'no_detections':[r for r in records if r['detections']==0],
    }
    rng=random.Random(812);selected=[]
    colors={'TP':'#008c70','FP':'#e68621','GT':'#2877d5','FN':'#c93570'}
    titles={'correct_localization':'Complete localization: no unmatched predictions or annotations', 'false_positives':'False-positive examples: orange boxes are unmatched predictions', 'missed_damage':'Missed-damage examples: magenta boxes are unmatched annotations', 'no_detections':'No detections at confidence > 0.25: magenta shows missed annotations'}
    for category,pool in pools.items():
        choices=rng.sample(pool,min(4,len(pool)))
        fig,axes=plt.subplots(len(choices),2,figsize=(12,3.8*len(choices)),squeeze=False,constrained_layout=True)
        for i,r in enumerate(choices):
            ps,pb,gb,mp,mg=details[r['image']]
            image_path=ROOT/'data/processed/cardd_yolo/images/val'/r['image']
            with Image.open(image_path) as image:im=image.convert('RGB')
            for ax in axes[i]:ax.imshow(im);ax.axis('off')
            for k,(x1,y1,x2,y2) in enumerate(gb):
                label='GT matched' if k in mg else 'FN'
                color=colors['GT' if k in mg else 'FN']
                axes[i,0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=2))
                axes[i,0].text(x1,y1,label,color='white',fontsize=8,bbox=dict(facecolor=color,alpha=.85,pad=1))
            for j,(x1,y1,x2,y2) in enumerate(pb):
                label='TP' if j in mp else 'FP';color=colors[label]
                axes[i,1].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=2))
                axes[i,1].text(x1,y1,f'{label} {ps[j]["score"]:.2f}',color='white',fontsize=8,bbox=dict(facecolor=color,alpha=.85,pad=1))
            axes[i,0].set_title(r['image']+' · ground truth',fontsize=10)
            axes[i,1].set_title(f'Predictions · TP {r["tp"]} / FP {r["fp"]} / FN {r["fn"]}',fontsize=10)
            selected.append(dict(category=category,**r,image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()))
        fig.suptitle(titles[category]+'\nValidation only · confidence > 0.25 · matching IoU > 0.45',fontsize=13)
        fig.savefig(out/(category+'.png'),dpi=150);plt.close(fig)
    with (out/'per_image.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    (out/'selection.json').write_text(json.dumps(dict(seed=812,sampling='Four random images per category from filename-sorted pools; categories may overlap.',pool_sizes={k:len(v) for k,v in pools.items()},selected=selected),indent=2))
    (out/'config.json').write_text(json.dumps(dict(source_audit=str(audit.relative_to(ROOT)),predictions=str(source.relative_to(ROOT)),predictions_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),confidence=.25,matching_iou=.45,seed=812,split='val',images=len(records)),indent=2))
    totals={k:sum(r[k] for r in records) for k in ['tp','fp','fn']}
    expected=next(r for r in json.loads((audit/'summary.json').read_text()) if r['confidence']==.25 and r['matching_iou']==.45)
    assert all(totals[k]==expected[k] for k in totals)
    (out/'README.md').write_text('# Validation error-analysis figures\n\nFixed confidence > 0.25 and matching IoU > 0.45, used for inspection only; no operational threshold is selected. Predictions are frozen post-NMS outputs from step 1. Rounded export coordinates/scores limit exact equivalence to in-memory evaluation.\n\nEach figure shows four reproducibly sampled validation images, with ground truth on the left and predictions on the right. Blue: matched annotation; magenta: missed annotation; green: matched prediction; orange: unmatched prediction. Matching all damage annotations does not imply a correct severity classification. Categories overlap and these illustrative samples do not estimate error prevalence.\n\nNo-detection cases must remain in the future severity evaluation through a predefined fallback rather than being discarded. No training, new inference, or test-set evaluation was performed. Original images, labels and checkpoint remain unchanged.\n\nselection.json records pool sizes, filenames, seed and image hashes. per_image.csv records counts for all 810 validation images. Counts agree with the step-2 audit.\n')
    print('OUTPUT',out);print('POOLS', {k:len(v) for k,v in pools.items()});print('TOTALS',totals)

if __name__=='__main__':main()
