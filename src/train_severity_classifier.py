"""Fine-tune ResNet50 for 3-class severity on whole images or YOLO crops; selects on validation and never loads test images."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
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
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

LABELS=['01-minor','02-moderate','03-severe']
MEAN,STD=[0.485,0.456,0.406],[0.229,0.224,0.225]
CFG=dict(img=224,batch=32,head_epochs=5,finetune_epochs=25,head_lr=1e-3,backbone_lr=1e-4,weight_decay=1e-4,label_smoothing=.1,patience=7,
         weights='IMAGENET1K_V2',train_aug='RandomResizedCrop(224, scale 0.8-1.0), horizontal flip, rotation +-10, ColorJitter(0.2, 0.2)',
         eval_transform='Resize(256), CenterCrop(224)',selection='best validation macro-F1; early stopping patience 7 epochs')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Images(Dataset):
    def __init__(self,rows,transform):self.rows,self.transform=rows,transform
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image']) as image:im=image.convert('RGB')
        return self.transform(im),LABELS.index(r['label'])


def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    prec,rec,f1,sup=precision_recall_fscore_support(y,p,labels=[0,1,2],zero_division=0)
    return dict(n=int(len(y)),accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',labels=[0,1,2],zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y,p) if len(set(y))>1 else None,mae_ordinal=float(np.abs(y-p).mean()),
                severe_as_minor=int(((y==2)&(p==0)).sum()),minor_as_severe=int(((y==0)&(p==2)).sum()),
                per_class={l:dict(precision=float(prec[k]),recall=float(rec[k]),f1=float(f1[k]),support=int(sup[k])) for k,l in enumerate(LABELS)},
                confusion_matrix_rows_true_cols_pred=confusion_matrix(y,p,labels=[0,1,2]).tolist())


def run_epoch(model,loader,device,criterion,optimizer=None):
    training=optimizer is not None;total=0;loss_sum=0;ys=[];ps=[];probs=[]
    with torch.set_grad_enabled(training):
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            out=model(x);loss=criterion(out,y)
            if training:optimizer.zero_grad();loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(y);total+=len(y)
            ys+=y.tolist();ps+=out.argmax(1).tolist();probs+=out.softmax(1).tolist()
    return loss_sum/total,ys,ps,probs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['whole','crop'],required=True);ap.add_argument('--seed',type=int,required=True)
    args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    audit=sorted((ROOT/'outputs/experiments').glob('severity_data3a_audit_*'))[-1]
    crops_dir=sorted((ROOT/'outputs/experiments').glob('severity_crops_*'))[-1]
    crops={r['path']:r for r in csv.DictReader((crops_dir/'crops.csv').open())}
    rows={'training':[],'validation':[]}
    for r in csv.DictReader((audit/'manifest.csv').open()):
        if r['include']!='True' or r['split'] not in rows:continue  # test split is never read here
        c=crops[r['path']]
        rows[r['split']].append(dict(path=r['path'],label=r['label'],fallback=c['fallback']=='True',
                                     image=str(ROOT/r['path'] if args.variant=='whole' else crops_dir/c['crop'])))
    out=ROOT/'outputs/experiments'/(f'severity_resnet50_{args.variant}_seed{args.seed}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    train_tf=transforms.Compose([transforms.RandomResizedCrop(CFG['img'],scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),
                                 transforms.ColorJitter(.2,.2),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(CFG['img']),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    g=torch.Generator();g.manual_seed(args.seed)
    train_loader=DataLoader(Images(rows['training'],train_tf),batch_size=CFG['batch'],shuffle=True,generator=g,num_workers=0)
    val_loader=DataLoader(Images(rows['validation'],eval_tf),batch_size=CFG['batch'],shuffle=False,num_workers=0)
    model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc=nn.Linear(model.fc.in_features,len(LABELS))
    model.to(device)
    criterion=nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])
    backbone=[p for n,p in model.named_parameters() if not n.startswith('fc.')]
    meta=dict(variant=args.variant,seed=args.seed,device=device,manifest=str((audit/'manifest.csv').relative_to(ROOT)),manifest_sha256=sha256(audit/'manifest.csv'),
              crops=str((crops_dir/'crops.csv').relative_to(ROOT)),crops_sha256=sha256(crops_dir/'crops.csv'),train_images=len(rows['training']),
              val_images=len(rows['validation']),test_images_loaded=0,script_sha256=sha256(Path(__file__)),python=sys.version,platform=platform.platform(),
              packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','scikit-learn','numpy','pillow']},
              note='MPS kernels are not fully deterministic; seeds fix data order and initialisation only.',**CFG)
    (out/'config.json').write_text(json.dumps(meta,indent=2))
    history=[];best=dict(macro_f1=-1);stale=0;started=time.time()
    total_epochs=CFG['head_epochs']+CFG['finetune_epochs']
    for epoch in range(1,total_epochs+1):
        if epoch==1:
            # Phase 1: train only the new head; frozen backbone BatchNorm stays in eval mode.
            for p in backbone:p.requires_grad=False
            optimizer=torch.optim.AdamW(model.fc.parameters(),lr=CFG['head_lr'],weight_decay=CFG['weight_decay']);scheduler=None
        if epoch==CFG['head_epochs']+1:
            for p in backbone:p.requires_grad=True
            optimizer=torch.optim.AdamW([dict(params=backbone,lr=CFG['backbone_lr']),dict(params=model.fc.parameters(),lr=CFG['head_lr'])],weight_decay=CFG['weight_decay'])
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['finetune_epochs'])
        phase='head' if epoch<=CFG['head_epochs'] else 'finetune'
        model.train()
        if phase=='head':
            for m in model.modules():
                if isinstance(m,nn.BatchNorm2d):m.eval()
        tl,ty,tp,_=run_epoch(model,train_loader,device,criterion,optimizer)
        if scheduler:scheduler.step()
        model.eval()
        vl,vy,vp,vprob=run_epoch(model,val_loader,device,criterion)
        vm=metrics(vy,vp)
        history.append(dict(epoch=epoch,phase=phase,train_loss=tl,train_accuracy=accuracy_score(ty,tp),val_loss=vl,val_accuracy=vm['accuracy'],
                            val_macro_f1=vm['macro_f1'],seconds=round(time.time()-started,1)))
        print(json.dumps(history[-1]),flush=True)
        if vm['macro_f1']>best['macro_f1']:
            best=dict(epoch=epoch,macro_f1=vm['macro_f1'],ys=vy,ps=vp,probs=vprob);stale=0
            torch.save(dict(state_dict=model.state_dict(),labels=LABELS,epoch=epoch,variant=args.variant,seed=args.seed,mean=MEAN,std=STD),out/'best.pt')
        else:
            stale+=1
            if phase=='finetune' and stale>=CFG['patience']:break
    with (out/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    with (out/'val_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','pred','fallback']+[f'p_{l}' for l in LABELS])
        for r,y,p,pr in zip(rows['validation'],best['ys'],best['ps'],best['probs']):w.writerow([r['path'],LABELS[y],LABELS[p],r['fallback']]+[round(v,5) for v in pr])
    fb=np.array([r['fallback'] for r in rows['validation']]);ys,ps=np.array(best['ys']),np.array(best['ps'])
    result=dict(best_epoch=best['epoch'],epochs_run=len(history),minutes=round((time.time()-started)/60,1),
                validation=metrics(ys,ps),validation_detected=metrics(ys[~fb],ps[~fb]),validation_fallback=metrics(ys[fb],ps[fb]),
                majority_baseline_accuracy=max(np.bincount(ys,minlength=3))/len(ys))
    (out/'val_metrics.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    e=[h['epoch'] for h in history]
    axes[0].plot(e,[h['train_loss'] for h in history],label='train');axes[0].plot(e,[h['val_loss'] for h in history],label='validation');axes[0].set_title('Loss')
    axes[1].plot(e,[h['train_accuracy'] for h in history],label='train accuracy');axes[1].plot(e,[h['val_accuracy'] for h in history],label='val accuracy')
    axes[1].plot(e,[h['val_macro_f1'] for h in history],label='val macro-F1');axes[1].set_title('Accuracy / macro-F1')
    for ax in axes:ax.axvline(CFG['head_epochs']+.5,color='grey',linestyle=':');ax.axvline(best['epoch'],color='#c93570',linestyle='--',linewidth=1);ax.set_xlabel('epoch');ax.legend(fontsize=8)
    fig.suptitle(f'ResNet50 severity · {args.variant} · seed {args.seed} · dotted: unfreeze · dashed: selected epoch {best["epoch"]}')
    fig.savefig(out/'curves.png',dpi=130);plt.close(fig)
    (out/'README.md').write_text(f'# ResNet50 severity classifier ({args.variant}, seed {args.seed})\n\nInput: {"whole data3a images" if args.variant=="whole" else "YOLO union crops (whole-image fallback when no detection)"} from the filtered manifest. Training and validation splits only; no test image was loaded.\n\nImageNet ResNet50 (IMAGENET1K_V2), 3-class head. Phase 1 trains the head for {CFG["head_epochs"]} epochs with the backbone frozen; phase 2 fine-tunes all layers (backbone lr 1e-4, head 1e-3, cosine) with early stopping. best.pt is the epoch with the highest validation macro-F1, so validation metrics are optimistic for model selection.\n\nval_metrics.json also splits validation results into images where YOLO detected damage and fallback images. Same-image groups are identical across variants, so the breakdown is comparable.\n')
    print('OUTPUT',out);print(json.dumps({k:result[k] for k in ['best_epoch','epochs_run','minutes']}|{'val':{k:result['validation'][k] for k in ['accuracy','macro_f1','mae_ordinal','severe_as_minor','minor_as_severe']}}))

if __name__=='__main__':main()
