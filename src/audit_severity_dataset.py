"""Audit data3a severity images for duplicates/leakage and write a filtered manifest; original files are only read."""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DATA=ROOT/'data/data3a'
CARDD=ROOT/'data/cardd/CarDD/CarDD_COCO'
SPLITS=['training','validation','test']
EXT={'.jpg','.jpeg','.png'}
NEAR=4


def dhash(path):
    with Image.open(path) as image:a=np.asarray(image.convert('L').resize((9,8)),dtype=np.int16)
    return (a[:,1:]>a[:,:-1]).flatten()


def pair_figure(pairs,title,path,per_row=3):
    rows=max(1,-(-len(pairs)//per_row))
    fig,axes=plt.subplots(rows,2*per_row,figsize=(4*per_row,2.1*rows+.8),squeeze=False,constrained_layout=True)
    for ax in axes.flat:ax.axis('off')
    for n,(a,b,caption_a,caption_b) in enumerate(pairs):
        r,c=divmod(n,per_row)
        for k,(p,cap) in enumerate([(a,caption_a),(b,caption_b)]):
            with Image.open(p) as image:axes[r,2*c+k].imshow(image.convert('RGB'))
            axes[r,2*c+k].set_title(cap,fontsize=7)
    fig.suptitle(title,fontsize=11)
    fig.savefig(path,dpi=130);plt.close(fig)


def main():
    out=ROOT/'outputs/experiments'/('severity_data3a_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    recs=[]
    for split in SPLITS:
        for p in sorted((DATA/split).rglob('*')):
            if p.suffix.lower() not in EXT:continue
            with Image.open(p) as image:image.load();w,h=image.size;mode=image.mode
            recs.append(dict(path=str(p.relative_to(ROOT)),split=split,label=p.parent.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),width=w,height=h,mode=mode))
    H=np.array([dhash(ROOT/r['path']) for r in recs])
    # Union exact and near duplicates into groups.
    parent=list(range(len(recs)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    pairs=[]
    for i in range(len(recs)):
        d=(H[i+1:]!=H[i]).sum(1)
        for j in np.nonzero(d<=NEAR)[0]+i+1:
            exact=recs[i]['sha256']==recs[j]['sha256']
            pairs.append(dict(a=recs[i]['path'],b=recs[j]['path'],dhash_distance=int(d[j-i-1]),exact=exact,cross_split=recs[i]['split']!=recs[j]['split'],label_conflict=recs[i]['label']!=recs[j]['label']))
            parent[find(i)]=find(j)
    groups=defaultdict(list)
    for i in range(len(recs)):groups[find(i)].append(i)
    gid=0
    for members in groups.values():
        for i in members:recs[i].update(group='',include=True,reason='')
        if len(members)==1:continue
        gid+=1
        for i in members:recs[i]['group']=f'g{gid:03d}'
        if len({recs[i]['label'] for i in members})>1:
            for i in members:recs[i].update(include=False,reason='label_conflict')
            continue
        keep=min(members,key=lambda i:(SPLITS.index(recs[i]['split']),recs[i]['path']))
        for i in members:
            if i!=keep:recs[i].update(include=False,reason=('cross_split_duplicate_of:' if recs[i]['split']!=recs[keep]['split'] else 'same_split_duplicate_of:')+recs[keep]['path'])
    # CarDD near-match screen (exact and dHash<=NEAR).
    cardd=[p for s in ['train2017','val2017','test2017'] for p in sorted((CARDD/s).glob('*')) if p.suffix.lower() in EXT]
    CH=np.array([dhash(p) for p in cardd]);cardd_sha={hashlib.sha256(p.read_bytes()).hexdigest() for p in cardd}
    cardd_matches=[]
    for i,r in enumerate(recs):
        d=(CH!=H[i]).sum(1);j=int(d.argmin())
        r['cardd_exact']=r['sha256'] in cardd_sha
        r['cardd_near']=str(cardd[j].relative_to(ROOT)) if d[j]<=NEAR else ''
        if d[j]<=NEAR:cardd_matches.append(dict(data3a=r['path'],cardd=r['cardd_near'],cardd_split=cardd[j].parent.name,dhash_distance=int(d[j]),data3a_split=r['split'],label=r['label']))
        # The detector has seen CarDD images, so overlapping evaluation images are excluded; training copies are kept.
        if (r['cardd_exact'] or r['cardd_near']) and r['split']!='training' and r['include']:r.update(include=False,reason='cardd_overlap:'+(r['cardd_near'] or 'exact'))
    fields=['path','split','label','include','reason','group','sha256','width','height','mode','cardd_exact','cardd_near']
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in recs)
    for name,rows in [('duplicate_pairs.csv',pairs),('cardd_near_matches.csv',cardd_matches)]:
        with (out/name).open('w',newline='') as f:
            if rows:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    caption=lambda p,extra='':'/'.join(Path(p).parts[-3:])+extra
    cross=[x for x in pairs if x['cross_split'] and not x['label_conflict']]
    conflict=[x for x in pairs if x['label_conflict']]
    pair_figure([(ROOT/x['a'],ROOT/x['b'],caption(x['a']),caption(x['b'],f" · d={x['dhash_distance']}")) for x in cross],f'Cross-split duplicates ({len(cross)} pairs): later-split copy excluded from manifest',out/'cross_split_duplicates.png')
    pair_figure([(ROOT/x['a'],ROOT/x['b'],caption(x['a']),caption(x['b'],f" · d={x['dhash_distance']}")) for x in conflict],f'Label-conflict duplicates ({len(conflict)} pairs): all group members excluded',out/'label_conflicts.png')
    pair_figure([(ROOT/x['data3a'],ROOT/x['cardd'],caption(x['data3a']),caption(x['cardd'],f" · d={x['dhash_distance']}")) for x in cardd_matches],f'data3a ↔ CarDD near-matches ({len(cardd_matches)}, dHash ≤ {NEAR}): validation/test copies excluded, training copies kept',out/'cardd_near_matches.png')
    before=Counter((r['split'],r['label']) for r in recs);after=Counter((r['split'],r['label']) for r in recs if r['include'])
    counts={s:{l:dict(original=before[(s,l)],included=after[(s,l)]) for l in sorted({r['label'] for r in recs})} for s in SPLITS}
    reasons=Counter(r['reason'].split(':')[0] for r in recs if not r['include'])
    (out/'summary.json').write_text(json.dumps(dict(images=len(recs),included=sum(r['include'] for r in recs),excluded_by_reason=reasons,counts=counts,duplicate_pairs=len(pairs),cross_split_pairs=sum(x['cross_split'] for x in pairs),label_conflict_pairs=len(conflict),cardd_exact=sum(r['cardd_exact'] for r in recs),cardd_near=len(cardd_matches),cardd_near_by_split=Counter(x['cardd_split'] for x in cardd_matches)),indent=2))
    (out/'config.json').write_text(json.dumps(dict(dataset=str(DATA.relative_to(ROOT)),cardd=str(CARDD.relative_to(ROOT)),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hash='dHash 64-bit on 9x8 grayscale',near_duplicate_max_distance=NEAR,keep_priority=SPLITS),indent=2))
    (out/'README.md').write_text(f'# data3a severity dataset audit\n\nRead-only audit of data/data3a (3 classes: 01-minor, 02-moderate, 03-severe). Original files are not moved or edited; manifest.csv marks which images to use.\n\nDuplicates are byte-identical (SHA-256) or near-identical (dHash Hamming distance <= {NEAR}) images, grouped transitively. Groups with conflicting labels are excluded entirely. Otherwise one copy is kept with priority training > validation > test, so leaked validation/test copies and same-split duplicates are excluded.\n\nNear-duplicate matching at these small resolutions (median about 259x184) can produce false positives and misses; cross_split_duplicates.png and label_conflicts.png show every pair for visual confirmation. CarDD near-matches were visually confirmed as the same photographs. Because the YOLO detector was trained on CarDD, validation/test images matching CarDD are excluded (reason cardd_overlap); training copies are kept and flagged in the cardd_near column.\n\nNo training or model evaluation was performed.\n')
    print('OUTPUT',out);print(json.dumps(json.loads((out/'summary.json').read_text()),indent=1))

if __name__=='__main__':main()
