"""Analyze frozen validation predictions; never tune on test data or rerun training."""
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics.utils.metrics import ConfusionMatrix


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources = sorted((ROOT/'outputs/experiments').glob('yolo11n_baseline_val_*/validation/predictions.json'))
    if not sources:
        raise FileNotFoundError('Run validate_yolo_baseline.py first.')
    source = sources[-1]
    annotations = ROOT/'data/cardd/CarDD/CarDD_COCO/annotations/instances_val2017.json'
    coco = json.loads(annotations.read_text())
    predictions = json.loads(source.read_text())
    images = {im['id']: im for im in coco['images']}
    pred, gt = defaultdict(list), defaultdict(list)
    for p in predictions:
        if p['image_id'] not in images or p['category_id'] != 1:
            raise ValueError('Unexpected prediction image or category')
        pred[p['image_id']].append(p)
    # Use the same clipped, six-decimal YOLO labels as the original validator.
    label_dir = ROOT/'data/processed/cardd_yolo/labels/val'
    label_hash = hashlib.sha256()
    for image_id, im in images.items():
        label = label_dir/(Path(im['file_name']).stem+'.txt')
        label_hash.update(label.name.encode()); label_hash.update(label.read_bytes())
        for line in label.read_text().splitlines():
            cls, xc, yc, w, h = map(float,line.split())
            assert cls == 0
            gt[image_id].append({'bbox':[(xc-w/2)*im['width'],(yc-h/2)*im['height'],w*im['width'],h*im['height']]})
    out = ROOT/'outputs/experiments'/('yolo11n_threshold_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    config = dict(predictions=str(source.relative_to(ROOT)), predictions_sha256=digest(source),
                  annotations=str(annotations.relative_to(ROOT)), annotations_sha256=digest(annotations), yolo_validation_labels_sha256=label_hash.hexdigest(),
                  script_sha256=digest(Path(__file__)), split='val', images=len(images), ground_truth_boxes=len(coco['annotations']),
                  confidence_thresholds=[0.001,0.1,0.25,0.5], matching_ious=[0.45,0.5],
                  source_nms_iou=0.7, source_max_det=300,
                  method='Installed Ultralytics ConfusionMatrix; one-to-one confidence-ordered matching; strict score > threshold and IoU > threshold.',
                  limitation='Saved JSON rounds boxes to 3 decimals and confidence to 5 decimals. Thresholds operate on frozen post-NMS predictions, not fresh inference. No AP recalculation or threshold selection.')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    def boxes(rows):
        a=np.array([r['bbox'] for r in rows],dtype=np.float32).reshape(-1,4)
        a[:,2:]+=a[:,:2]
        return torch.from_numpy(a)
    summary, per_image = [], []
    for matching_iou in config['matching_ious']:
        for conf in config['confidence_thresholds']:
            matrix = np.zeros((2,2))
            no_detection = 0
            for image_id, im in images.items():
                ps=pred[image_id]; gs=gt[image_id]
                det=dict(cls=torch.zeros(len(ps)),conf=torch.tensor([p['score'] for p in ps]),bboxes=boxes(ps))
                batch=dict(cls=torch.zeros(len(gs)),bboxes=boxes(gs))
                cm=ConfusionMatrix(names={0:'damage'})
                cm.process_batch(det,batch,conf=conf,iou_thres=matching_iou)
                matrix+=cm.matrix
                n=sum(p['score']>conf for p in ps)
                no_detection+=n==0
                tp,fp,fn=int(cm.matrix[0,0]),int(cm.matrix[0,1]),int(cm.matrix[1,0])
                per_image.append(dict(image=im['file_name'],confidence=conf,matching_iou=matching_iou,detections=n,tp=tp,fp=fp,fn=fn))
            tp,fp,fn=int(matrix[0,0]),int(matrix[0,1]),int(matrix[1,0])
            assert tp+fn==len(coco['annotations'])
            summary.append(dict(confidence=conf,matching_iou=matching_iou,tp=tp,fp=fp,fn=fn,detections=tp+fp,
                                precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),
                                f1=2*tp/(2*tp+fp+fn),no_detection_images=no_detection))
    for name,rows in [('summary',summary),('per_image',per_image)]:
        (out/(name+'.json')).write_text(json.dumps(rows,indent=2))
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    for ax,r in zip(axes,summary[:4]):
        m=np.array([[r['tp'],r['fp']],[r['fn'],0]])
        ax.imshow(np.log1p(m),cmap='Blues',vmin=0,vmax=np.log1p(max(x['fp'] for x in summary)))
        for (i,j),v in np.ndenumerate(m):ax.text(j,i,f'{v:,}' if (i,j)!=(1,1) else 'N/A',ha='center',va='center',color='white' if v>10000 else '#172b2a')
        ax.set(xticks=[0,1],xticklabels=['Damage','Background'],yticks=[0,1],yticklabels=['Damage','Background'],title=f'Confidence > {r["confidence"]}',xlabel='Ground truth')
    axes[0].set_ylabel('Predicted')
    fig.suptitle('Validation confusion matrices · matching IoU > 0.45 · color uses log(1 + count)')
    fig.savefig(out/'confusion_threshold_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    scores=np.array([p['score'] for p in predictions])
    axes[0].hist(scores,bins=np.linspace(0,1,51),color='#08796b');axes[0].set(yscale='log',xlabel='Confidence',ylabel='Prediction count (log scale)',title='Frozen post-NMS prediction scores')
    for iou in config['matching_ious']:
        rows=[r for r in summary if r['matching_iou']==iou]
        axes[1].plot([r['confidence'] for r in rows],[r['precision'] for r in rows],'o-',label=f'Precision, IoU {iou}')
        axes[1].plot([r['confidence'] for r in rows],[r['recall'] for r in rows],'s--',label=f'Recall, IoU {iou}')
    axes[1].set(xlabel='Confidence threshold',ylabel='Score',ylim=(0,1),title='Precision–recall tradeoff');axes[1].legend(fontsize=8)
    fig.savefig(out/'confidence_analysis.png',dpi=180);plt.close(fig)
    prior=json.loads((source.parent.parent/'metrics.json').read_text())
    initial=summary[0]
    report=['# Validation confidence-threshold audit','',config['method'],'',config['limitation'],'',
            'All six CarDD damage types are treated as damage. No severity labels are inferred. No training or test-set evaluation occurred.',
            '',f'At confidence 0.001 and matching IoU 0.45: TP={initial["tp"]}, FP={initial["fp"]}, FN={initial["fn"]}.',
            f'Source in-memory confusion matrix: {prior["confusion_matrix"]}. Differences arise from rounded JSON scores/coordinates and reconstruction of label coordinates. Scores rounded to exactly 0.001 are excluded by the strict filter; this export cannot exactly reconstruct the in-memory matrix.',
            '', 'The low-confidence matrix and the headline validation precision/recall measure different operating points. AP uses ranked predictions across confidence thresholds; these fixed-threshold precision/recall values are not AP.',
            'Background in a detection confusion matrix represents unmatched boxes or annotations. The bottom-right cell is not a true-negative count.',
            'The validation split contains no fully unannotated background images; this analysis does not establish performance on an undamaged-vehicle dataset.',
            '', 'No operational threshold is selected here. Choose a crop-generation policy using validation data only after reviewing errors and no-detection cases.']
    (out/'README.md').write_text('\n'.join(report)+'\n')
    print('Saved:',out)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
