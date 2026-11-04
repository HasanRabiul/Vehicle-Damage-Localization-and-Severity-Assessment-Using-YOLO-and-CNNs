"""Evaluate severity runs on validation (default) or, only with --split test, the test split: seed aggregates, input cross-check, bootstrap CIs and error panels."""
import argparse
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
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models,transforms

from train_severity_classifier import LABELS,MEAN,STD,Images,metrics

VARIANTS=['whole','crop']
SEEDS=[812,813,814]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_runs():
    runs={}
    for d in sorted((ROOT/'outputs/experiments').glob('severity_resnet50_*_seed*_*')):
        if not (d/'best.pt').exists() or not (d/'val_metrics.json').exists():continue
        variant,seed=d.name.split('_')[2],int(d.name.split('_')[3][4:])
        runs[(variant,seed)]=d  # sorted by timestamp, so the latest run wins
    missing=[k for k in [(v,s) for v in VARIANTS for s in SEEDS] if k not in runs]
    assert not missing,f'missing runs: {missing}'
    return runs


def bootstrap(y,pred_sets,n=2000,seed=812):
    """95% percentile CIs of seed-mean macro-F1/accuracy per configuration, resampling images; pred_sets maps name -> list of per-seed predictions."""
    rng=np.random.default_rng(seed);draws=defaultdict(list)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        for name,ps in pred_sets.items():
            draws[name+'/macro_f1'].append(np.mean([metrics(y[idx],p[idx])['macro_f1'] for p in ps]))
            draws[name+'/accuracy'].append(np.mean([(y[idx]==p[idx]).mean() for p in ps]))
    ci=lambda a:dict(low=float(np.percentile(a,2.5)),high=float(np.percentile(a,97.5)))
    out={k:ci(v) for k,v in draws.items()}
    diff=np.array(draws['whole->whole/macro_f1'])-np.array(draws['crop->crop/macro_f1'])
    out['difference_whole_minus_crop/macro_f1']=dict(**ci(diff),fraction_draws_at_or_below_zero=float((diff<=0).mean()))
    return dict(resamples=n,seed=seed,method='paired percentile bootstrap over images; statistic is the mean over seeds',intervals=out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['validation','test'],default='validation');args=ap.parse_args()
    audit=sorted((ROOT/'outputs/experiments').glob('severity_data3a_audit_*'))[-1]
    crops_dir=sorted((ROOT/'outputs/experiments').glob('severity_crops_*'))[-1]
    crops={r['path']:r for r in csv.DictReader((crops_dir/'crops.csv').open())}
    val=[r for r in csv.DictReader((audit/'manifest.csv').open()) if r['include']=='True' and r['split']==args.split]
    inputs={v:[dict(path=r['path'],label=r['label'],fallback=crops[r['path']]['fallback']=='True',
                    image=str(ROOT/r['path'] if v=='whole' else crops_dir/crops[r['path']]['crop'])) for r in val] for v in VARIANTS}
    out=ROOT/'outputs/experiments'/(f'severity_eval_{args.split}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    runs=latest_runs();rows=[];preds={};reproduction={}
    y=np.array([LABELS.index(r['label']) for r in val]);fb=np.array([r['fallback'] for r in inputs['whole']])
    for (trained_on,seed),run in sorted(runs.items()):
        model=models.resnet50(weights=None);model.fc=nn.Linear(model.fc.in_features,len(LABELS))
        model.load_state_dict(torch.load(run/'best.pt',map_location='cpu')['state_dict']);model.to(device).eval()
        for evaluated_on in VARIANTS:
            probs=[]
            with torch.no_grad():
                for x,_ in DataLoader(Images(inputs[evaluated_on],tf),batch_size=64,shuffle=False):probs+=model(x.to(device)).softmax(1).cpu().tolist()
            probs=np.array(probs);p=probs.argmax(1);preds[(trained_on,seed,evaluated_on)]=(p,probs)
            for subset,mask in [('all',np.ones_like(fb)),('detected',~fb),('fallback',fb)]:
                m=metrics(y[mask],p[mask])
                rows.append(dict(trained_on=trained_on,evaluated_on=evaluated_on,seed=seed,subset=subset,n=m['n'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],
                                 balanced_accuracy=m['balanced_accuracy'],mae_ordinal=m['mae_ordinal'],severe_as_minor=m['severe_as_minor'],minor_as_severe=m['minor_as_severe'],
                                 **{f'recall_{l}':m['per_class'][l]['recall'] for l in LABELS},run=run.name))
        if args.split!='validation':continue
        # The matched-input result should reproduce the training script's saved validation score.
        saved=json.loads((run/'val_metrics.json').read_text())['validation']['macro_f1']
        again=next(r['macro_f1'] for r in rows if r['run']==run.name and r['evaluated_on']==trained_on and r['subset']=='all')
        reproduction[run.name]=dict(saved=saved,re_evaluated=again,difference=round(again-saved,4))
        print(run.name,'saved',round(saved,4),'re-evaluated',round(again,4),flush=True)
    with (out/'per_run.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    agg=defaultdict(dict);keys=['accuracy','macro_f1','balanced_accuracy','mae_ordinal','severe_as_minor','minor_as_severe']+[f'recall_{l}' for l in LABELS]
    for t in VARIANTS:
        for e in VARIANTS:
            for subset in ['all','detected','fallback']:
                rs=[r for r in rows if r['trained_on']==t and r['evaluated_on']==e and r['subset']==subset]
                agg[f'{t}->{e}'][subset]=dict(n=rs[0]['n'],seeds=len(rs),**{k:dict(mean=float(np.mean([r[k] for r in rs])),std=float(np.std([r[k] for r in rs],ddof=1))) for k in keys})
    baseline=float(max(np.bincount(y,minlength=3))/len(y))
    (out/'summary.json').write_text(json.dumps(dict(split=args.split,images=len(val),majority_baseline_accuracy=baseline,chance_macro_f1_approx=1/3,
                                                    std='sample std over 3 seeds',reproduction_macro_f1=reproduction or None,results=agg,
                                                    bootstrap=bootstrap(y,{f'{v}->{v}':[preds[(v,s,v)][0] for s in SEEDS] for v in VARIANTS})),indent=2))
    # 2x2 macro-F1 heat table.
    fig,ax=plt.subplots(figsize=(5.2,4),constrained_layout=True)
    M=np.array([[agg[f'{t}->{e}']['all']['macro_f1']['mean'] for e in VARIANTS] for t in VARIANTS])
    S=np.array([[agg[f'{t}->{e}']['all']['macro_f1']['std'] for e in VARIANTS] for t in VARIANTS])
    ax.imshow(M,cmap='Blues',vmin=.4,vmax=.8)
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{M[i,j]:.3f}\n± {S[i,j]:.3f}',ha='center',va='center',fontsize=12,color='white' if M[i,j]>.65 else 'black')
    ax.set_xticks([0,1],[f'evaluated on {v}' for v in VARIANTS]);ax.set_yticks([0,1],[f'trained on {v}' for v in VARIANTS])
    ax.set_title(f'{args.split.capitalize()} macro-F1 (mean ± std, 3 seeds)',fontsize=11)
    fig.savefig(out/'cross_input_macro_f1.png',dpi=140);plt.close(fig)
    # Seed-summed confusion matrices for matched inputs.
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    for ax,v in zip(axes,VARIANTS):
        cm=sum(np.array(metrics(y,preds[(v,s,v)][0])['confusion_matrix_rows_true_cols_pred']) for s in SEEDS)
        ax.imshow(cm,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,int(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS);ax.set_xlabel('predicted');ax.set_ylabel('true');ax.set_title(f'{v} model on {v} input (3 seeds summed)')
    fig.savefig(out/'confusion_matrices.png',dpi=140);plt.close(fig)
    # Error panels (seed 812, matched input): three random errors per true class, plus one correct example per class for reference.
    rng=random.Random(812);selected=[]
    for v in VARIANTS:
        p,probs=preds[(v,812,v)]
        fig,axes=plt.subplots(3,4,figsize=(13,8.5),constrained_layout=True)
        for c in range(3):
            wrong=[i for i in range(len(y)) if y[i]==c and p[i]!=c];right=[i for i in range(len(y)) if y[i]==c and p[i]==c]
            pick=rng.sample(right,min(1,len(right)))+rng.sample(wrong,min(3,len(wrong)))
            for k,ax in enumerate(axes[c]):
                ax.axis('off')
                if k>=len(pick):continue
                i=pick[k]
                with Image.open(inputs[v][i]['image']) as image:ax.imshow(image.convert('RGB'))
                ok=p[i]==c
                ax.set_title(f"{'correct' if ok else 'WRONG'} · true {LABELS[c][3:]} → {LABELS[p[i]][3:]} ({probs[i,p[i]]:.2f})"+(' · fallback' if fb[i] else ''),fontsize=8,color='#008c70' if ok else '#c93570')
                selected.append(dict(variant=v,path=inputs[v][i]['path'],true=LABELS[c],pred=LABELS[p[i]],confidence=round(float(probs[i,p[i]]),4),fallback=bool(fb[i])))
        fig.suptitle(f'{v} model, seed 812, {args.split} · first column correct, others errors (input as the model sees it before resizing)',fontsize=11)
        fig.savefig(out/f'errors_{v}.png',dpi=130);plt.close(fig)
    (out/'selection.json').write_text(json.dumps(dict(seed=812,selected=selected),indent=2))
    (out/'config.json').write_text(json.dumps(dict(split=args.split,test_images_loaded=len(val) if args.split=='test' else 0,device=device,manifest=str((audit/'manifest.csv').relative_to(ROOT)),
        crops=str((crops_dir/'crops.csv').relative_to(ROOT)),runs={f'{v}_seed{s}':str(d.relative_to(ROOT)) for (v,s),d in runs.items()},
        checkpoints_sha256={f'{v}_seed{s}':sha256(d/'best.pt') for (v,s),d in runs.items()},script_sha256=sha256(Path(__file__)),
        eval_transform='Resize(256), CenterCrop(224)'),indent=2))
    (out/'README.md').write_text(f'# Severity classifier {args.split} evaluation\n\nAll six best.pt checkpoints (whole/crop x seeds 812-814) evaluated on the {len(val)} filtered {args.split} images with each input type. Diagonal cells (trained and evaluated on the same input) are the planned comparison; off-diagonal cells measure input mismatch. detected/fallback subsets split by whether YOLO found damage. summary.json bootstrap gives paired 95% CIs (2,000 image resamples) for the seed-mean of the two planned configurations and their difference.\n\n'+('Validation was also used for epoch selection, so these numbers are optimistic. No test image was loaded.' if args.split=='validation' else 'Single pre-approved test evaluation; configurations and checkpoints were fixed on validation beforehand and nothing is re-tuned from these results.')+' Error panels are illustrative random samples (seed 812), not prevalence estimates.\n')
    print('OUTPUT',out)
    print(json.dumps(json.loads((out/'summary.json').read_text())['bootstrap']['intervals'],indent=1))
    for k,v in agg.items():print(k,{s:f"F1 {v[s]['macro_f1']['mean']:.3f}±{v[s]['macro_f1']['std']:.3f} acc {v[s]['accuracy']['mean']:.3f} MAE {v[s]['mae_ordinal']['mean']:.3f}" for s in v})

if __name__=='__main__':main()
