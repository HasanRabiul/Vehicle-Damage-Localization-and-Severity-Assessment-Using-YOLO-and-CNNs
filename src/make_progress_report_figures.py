"""Build the progress-report figures from saved experiment outputs; experiment folders are only read."""
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP=ROOT/'outputs/experiments'
OUT=ROOT/'reports/progress/figures'
WHOLE,CROP='#2a78d6','#eb6834'  # categorical slots 1 and 2
INK,MUTED,GRID='#1f1f1e','#6b6a63','#e4e3dd'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10,'axes.edgecolor':MUTED,
                     'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.color':GRID,'grid.linewidth':.6,'axes.axisbelow':True,'savefig.dpi':220,'savefig.bbox':'tight'})


def latest(pattern):
    return sorted(EXP.glob(pattern))[-1]


def detector_training():
    rows=[{k.strip():float(v) for k,v in r.items()} for r in csv.DictReader((ROOT/'runs/detect/runs/damage_detection/yolo11n_initial/results.csv').open())]
    e=[r['epoch'] for r in rows]
    fig,ax=plt.subplots(figsize=(5.4,2.9))
    for key,label,c in [('metrics/mAP50(B)','mAP50',WHOLE),('metrics/mAP50-95(B)','mAP50–95',CROP)]:
        y=[r[key] for r in rows]
        ax.plot(e,y,color=c,lw=2,marker='o',ms=4)
        ax.annotate(f'{label} {y[-1]:.3f}',(e[-1],y[-1]),xytext=(6,0),textcoords='offset points',va='center',color=INK,fontsize=9)
    ax.set(xlabel='Epoch',ylabel='Validation score',xlim=(.5,12.2),ylim=(0,.6),xticks=e)
    ax.legend(['mAP50','mAP50–95'],frameon=False,loc='upper left')
    fig.savefig(OUT/'detector_training.png');plt.close(fig)


def recall_by_damage_count():
    rows=list(csv.DictReader((latest('yolo11n_error_analysis_*')/'per_image.csv').open()))
    g=defaultdict(lambda:[0,0,0])
    for r in rows:
        k=min(int(r['ground_truth']),5);g[k][0]+=int(r['tp']);g[k][1]+=int(r['ground_truth']);g[k][2]+=1
    ks=sorted(g);rec=[g[k][0]/g[k][1] for k in ks]
    fig,ax=plt.subplots(figsize=(5.4,2.7))
    bars=ax.bar([str(k) if k<5 else '5+' for k in ks],rec,color=WHOLE,width=.6)
    for b,k,v in zip(bars,ks,rec):
        ax.text(b.get_x()+b.get_width()/2,v+.02,f'{v:.0%}',ha='center',color=INK,fontsize=9)
        ax.text(b.get_x()+b.get_width()/2,.03,f'n={g[k][2]}',ha='center',color='white',fontsize=8)
    ax.set(xlabel='Annotated damaged areas in the image',ylabel='Recall',ylim=(0,1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis='x',visible=False)
    fig.savefig(OUT/'recall_by_damage_count.png');plt.close(fig)
    return {('5+' if k==5 else str(k)):dict(images=g[k][2],recall=g[k][0]/g[k][1]) for k in ks}


def severity_comparison():
    s=json.loads((latest('severity_eval_test_*')/'summary.json').read_text())
    v=json.loads((latest('severity_eval_validation_*')/'summary.json').read_text())
    ci=s['bootstrap']['intervals']
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw=dict(width_ratios=[1,1.35],wspace=.35))
    for i,(name,key,c) in enumerate([('Whole image','whole',WHOLE),('Localized\n(YOLO crop)','crop',CROP)]):
        m=s['results'][f'{key}->{key}']['all']['macro_f1']['mean'];lo,hi=ci[f'{key}->{key}/macro_f1']['low'],ci[f'{key}->{key}/macro_f1']['high']
        a.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',color=c,ms=8,capsize=5,lw=2)
        a.text(i+.12,m,f'{m:.3f}',va='center',color=INK,fontsize=9)
        vm=v['results'][f'{key}->{key}']['all']['macro_f1']['mean']
        a.plot(i,vm,marker='D',ms=5,mfc='white',mec=c,mew=1.5)
    a.plot([],[],'o',color=MUTED,label='Test (95% CI)');a.plot([],[],'D',mfc='white',mec=MUTED,label='Validation')
    a.legend(frameon=False,fontsize=8,loc='lower left')
    a.set(xticks=[0,1],xticklabels=['Whole image','Localized\n(YOLO crop)'],xlim=(-.5,1.6),ylim=(.4,.75),ylabel='Macro-F1')
    a.set_title('(a) Macro-F1, mean of 3 seeds',fontsize=10,color=INK)
    classes=['01-minor','02-moderate','03-severe'];x=np.arange(3);w=.36
    for off,key,c,label in [(-w/2,'whole',WHOLE,'Whole image'),(w/2,'crop',CROP,'Localized (YOLO crop)')]:
        vals=[s['results'][f'{key}->{key}']['all'][f'recall_{k}']['mean'] for k in classes]
        bars=b.bar(x+off,vals,w-.04,color=c,label=label)
        for bar,val in zip(bars,vals):b.text(bar.get_x()+bar.get_width()/2,val+.015,f'{val:.2f}',ha='center',fontsize=8,color=INK)
    b.set(xticks=x,xticklabels=['Minor','Moderate','Severe'],ylim=(0,1),ylabel='Recall')
    b.grid(axis='x',visible=False);b.legend(frameon=False,fontsize=8,loc='upper right')
    b.set_title('(b) Test recall by class',fontsize=10,color=INK)
    fig.savefig(OUT/'severity_comparison.png');plt.close(fig)


def timeline():
    tasks=[('Stronger detector','2026-11-02','2026-11-10'),('Literature review','2026-11-02','2026-11-22'),('Localization variants','2026-11-09','2026-11-17'),
           ('Final configurations fixed','2026-11-18','2026-11-18'),('Final test comparison','2026-11-19','2026-11-21'),
           ('Moderate-class and error analysis','2026-11-19','2026-11-28'),('Demo','2026-11-23','2026-12-01'),('Final report and slides','2026-11-26','2026-12-06')]
    d=lambda s:np.datetime64(s)
    fig,ax=plt.subplots(figsize=(7,3))
    for i,(name,a,b) in enumerate(tasks):
        if a==b:ax.plot(d(a),i,marker='D',ms=9,color=CROP,zorder=3);ax.axvline(d(a),color=CROP,ls='--',lw=1,zorder=1)
        else:ax.barh(i,(d(b)-d(a)).astype(int)+1,left=d(a),height=.5,color=WHOLE,alpha=.85)
    ax.set_yticks(range(len(tasks)),[t[0] for t in tasks]);ax.invert_yaxis()
    ticks=np.arange(d('2026-11-02'),d('2026-12-08'),7);ax.set_xticks(ticks,[f"{str(t)[5:7]=='11' and 'Nov' or 'Dec'} {int(str(t)[8:])}" for t in ticks])
    ax.grid(axis='y',visible=False);ax.set_xlabel('Week starting (2026, target dates)')
    fig.savefig(OUT/'timeline.png');plt.close(fig)


def copy_existing():
    picks={
        'threshold_audit.png':latest('yolo11n_threshold_audit_*')/'confidence_analysis.png',
        'detector_missed_damage.png':latest('yolo11n_error_analysis_*')/'missed_damage.png',
        'detector_false_positives.png':latest('yolo11n_error_analysis_*')/'false_positives.png',
        'audit_cross_split.png':latest('severity_data3a_audit_*')/'cross_split_duplicates.png',
        'crop_samples.png':latest('severity_crops_*')/'crop_samples.png',
        'curves_whole_seed812.png':latest('severity_resnet50_whole_seed812_*')/'curves.png',
        'curves_crop_seed812.png':latest('severity_resnet50_crop_seed812_*')/'curves.png',
        'test_confusion.png':latest('severity_eval_test_*')/'confusion_matrices.png',
        'test_errors_whole.png':latest('severity_eval_test_*')/'errors_whole.png',
    }
    for name,src in picks.items():shutil.copy2(src,OUT/name)
    return {k:str(v.relative_to(ROOT)) for k,v in picks.items()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    detector_training();counts=recall_by_damage_count();severity_comparison();timeline();sources=copy_existing()
    (OUT/'sources.json').write_text(json.dumps(dict(copied=sources,recall_by_damage_count=counts),indent=2))
    print(json.dumps(counts,indent=1));print(sorted(p.name for p in OUT.glob('*.png')))

if __name__=='__main__':main()
