// ---------- Metrics explained ----------
// Every number on this page is read from the saved outputs (findings.py). The confusion matrix and the bootstrap
// histogram are recomputed in the browser from the saved final test predictions, which reproduce the saved macro-F1.
const MC={p:'#2a78d6',r:'#eb6834',f:'#067568'},SVGNS='http://www.w3.org/2000/svg';
function svgEl(tag,a,parent,text){const e=document.createElementNS(SVGNS,tag);for(const k in a)e.setAttribute(k,a[k]);if(text!==undefined)e.textContent=text;if(parent)parent.append(e);return e}
const f2=v=>v.toFixed(2),n0=v=>Math.round(v).toLocaleString();

function renderMeasures(F){
  const host=$('#measuresbody'),jump=$('#measuresjump');host.replaceChildren();jump.replaceChildren();
  const test=F.detector_test,det=test&&test.results.find(r=>r.role==='chosen'),sel=F.detector_selection,six=F.six_class_test,sixv=F.six_class_val,
    sev=F.severity_test||[],fin=sev.find(s=>s.final_comparison),r1=sev[0]!==fin?sev[0]:null,ea=F.error_analysis,val=F.severity_validation,
    cfg=F.final_configs,chosenCrop=(F.crop_variants||[]).find(v=>v.chosen),round1Crop=(F.crop_variants||[]).find(v=>v.round1),P=F.severity_predictions;
  if(!det||!fin||!ea){host.append(el('p','muted','The final outputs are not all available yet, so the metric examples cannot be filled in.'));return}
  const sub=Object.fromEntries(ea.subsets.map(s=>[s.subset,s])),pc=(v,l)=>ea.per_class.find(r=>r.input===v&&r.label===l),
    gt=det.tp+det.fn,sixc=six&&six.results.yolo11s_6cls_e100,sixb=sixc&&sixc.best,fc=fin.final_comparison;

  // Concept map: how the measures feed each other, lane by lane.
  const map=el('article','mmap');map.append(el('h2',null,'How the measures connect'));
  const lanes=[['Detection · CarDD',[['iou','IoU'],['conf','Confidence & NMS'],['counts','TP · FP · FN'],['pr','Precision ↔ Recall'],['f1','F1 → threshold'],['ap','AP → mAP50 / 50–95'],['coverage','Recall per image']]],
    ['Severity · data3a',[['confusion','Confusion matrix'],['accuracy','Accuracy'],['perclass','Per-class P · R · F1'],['macro','Macro-F1'],['deltas','ΔF1 · ΔR moderate'],['softmax','Softmax probability']]],
    ['Statistics · how sure',[['seeds','Seeds & variance'],['bootstrap','Paired bootstrap → 95% CI'],['splits','Validation vs test'],['posthoc','Post-hoc subsets'],['gradcam','Grad-CAM ratio'],['croparea','Crop area & padding']]]];
  for(const [name,items] of lanes){const lane=el('div','mlane');lane.append(el('b',null,name));const row=el('div','mchain');items.forEach(([id,t],i)=>{if(i)row.append(el('span','marrow','→'));const a=el('a',null,t);a.href='#m-'+id;a.onclick=e=>{e.preventDefault();document.getElementById('m-'+id).scrollIntoView({behavior:'smooth'})};row.append(a)});lane.append(row);map.append(lane)}
  map.append(el('p','muted','Arrows read “is built from” in reverse: IoU decides which boxes count as hits, the hit counts give precision and recall, F1 combines them to pick the confidence threshold, and AP summarises the whole precision–recall curve. On the severity side the confusion matrix gives per-class precision and recall, their F1 scores average into macro-F1, and the paired bootstrap says how sure we can be about the difference between whole-image and localized inputs.'));
  host.append(map);

  let group;const groupHead=(t,lead)=>{group=el('section','mgroup');const h=el('div','fighead');h.append(el('h2',null,t));group.append(h,el('p','figtext',lead));host.append(group);const j=el('a',null,t);j.href='#';j.onclick=e=>{e.preventDefault();h.scrollIntoView({behavior:'smooth'})};jump.append(j)};
  // A card: left column explains the measure, right column shows its role in this project with real numbers.
  const card=(id,tag,title,what,role)=>{const a=el('article','mcard');a.id='m-'+id;const h=el('div','fighead');h.append(el('span','stage',tag),el('h2',null,title));a.append(h);const g=el('div','mgrid'),l=el('div','mwhat'),r=el('div','mrole');l.innerHTML=what;r.innerHTML='<b class="mtag">In this project</b>'+role;g.append(l,r);a.append(g);group.append(a);return a};
  const fx=t=>`<div class="formula">${t}</div>`;

  // ===== Detection =====
  groupHead('Detection measures','The YOLO detector draws boxes around damage on CarDD images. These measures say how many real damages it finds and how many of its boxes are real. They are all computed against the hand-drawn CarDD boxes.');

  const iou=card('iou','DETECTION · BUILDING BLOCK','Intersection over Union (IoU)',
    `<p>IoU measures how well a predicted box overlaps a true (ground-truth) box: the overlapping area divided by the area the two boxes cover together. It is 0 when the boxes do not touch and 1 when they are identical.</p>${fx('IoU = area(predicted ∩ true) ÷ area(predicted ∪ true)')}<p>Drag the slider to move the predicted box and watch when it stops counting as a hit.</p><div id="ioudemo" class="ioudemo"></div>`,
    `<ul><li><b>Matching rule.</b> A predicted box finds a damage only if its IoU with an unmatched true box is at least <b>0.5</b>. This rule is behind every precision, recall and F1 on this page.</li>
    <li><b>mAP50 vs mAP50–95.</b> mAP50 uses the same 0.5 rule. mAP50–95 averages over the stricter IoU thresholds 0.50, 0.55 … 0.95, so it rewards tight boxes. The chosen detector scores <b>${f3(det.mAP50)}</b> vs <b>${f3(det.mAP50_95)}</b> on the CarDD test: it usually finds the damage, but its edges are loose. For this project that matters less, because the crop is the union of the boxes padded by ${(F.models.pad*100).toFixed(0)}% per side anyway.</li>
    <li><b>NMS IoU 0.7.</b> When two predicted boxes overlap by more than 0.7, non-maximum suppression keeps only the higher-scoring one, so one damage is not counted twice. The Test model tab exposes it as a slider.</li></ul>`);
  iouDemo(iou.querySelector('#ioudemo'));

  card('conf','DETECTION · BUILDING BLOCK','Confidence score, threshold and NMS',
    `<p>Every box YOLO predicts comes with a <b>confidence score</b> between 0 and 1. A <b>confidence threshold</b> keeps only the boxes that score at least that value. A low threshold keeps many boxes, including wrong ones; a high threshold keeps only the boxes the model is sure about, and misses more damage.</p><p>The score ranks boxes. It is not a calibrated probability: a box with 0.8 is not “right 80% of the time”.</p>`,
    `<ul><li>The frozen detector uses <b>confidence ${f2(cfg.detector.confidence)}</b>, chosen on CarDD validation by the F1 rule (see F1 below). Round 1 used YOLO11n at ${round1Crop?f2(round1Crop.confidence):'0.25'}.</li>
    <li>The threshold decides <b>which boxes build the localized crop</b>. When no box reaches it, the image falls back to the whole image: <b>${sub.fallback.n} of ${sub.all.n}</b> data3a test images (${pct(sub.fallback.n/sub.all.n)}) fell back. On the CarDD test, only ${det.no_detection_images} of ${test.images} images had no detection, so the fallback rate shows how different data3a photos are from CarDD.</li>
    <li>In the Test model tab the slider changes only the displayed boxes. The crop always uses the frozen ${f2(cfg.detector.confidence)}.</li></ul>`);

  card('counts','DETECTION · COUNTING','True positives, false positives, false negatives',
    `<p>After matching, every box ends up in one of three counts:</p><ul><li><b>TP (true positive):</b> a predicted box matched to a real damage (IoU ≥ 0.5).</li><li><b>FP (false positive):</b> a predicted box with no match. It is in the wrong place, or it is a second box on a damage that was already matched.</li><li><b>FN (false negative):</b> a real damage that no predicted box matched.</li></ul><p><b>Matching is greedy and one-to-one.</b> Predictions are taken from the highest confidence down, and each one claims the unmatched true box it overlaps most, if that IoU is ≥ 0.5. A true box can be claimed only once.</p><p class="note">There are no <b>true negatives</b> in detection: you cannot count “every patch of background without a box”. That is why detectors are scored with precision and recall, not accuracy.</p>`,
    `<p>CarDD test, chosen detector at ${f2(det.confidence)}, ${n0(gt)} true damage boxes in ${test.images} images:</p><div class="mcounts"><div><small>TP</small><strong>${det.tp}</strong><span>damages found</span></div><div><small>FP</small><strong>${det.fp}</strong><span>boxes on nothing</span></div><div><small>FN</small><strong>${det.fn}</strong><span>damages missed</span></div></div>
    <p>TP + FN = ${det.tp} + ${det.fn} = ${n0(gt)}, which is every true box.</p>${sixb?`<p>For the six-class study the damage type is ignored when matching (<b>class-agnostic</b> evaluation). This counts the one-class and six-class detectors the same way: the six-class model gets ${sixb.tp} / ${sixb.fp} / ${sixb.fn} on the same test images.</p>`:''}`);

  const pr=card('pr','DETECTION · CORE PAIR','Precision and recall, and how they trade off',
    `<p><b>Precision</b> asks: of the boxes I drew, how many are real damage? <b>Recall</b> asks: of the real damage, how much did I find? Both share TP on top. Precision is hurt by false alarms (FP); recall is hurt by misses (FN).</p>${fx(`Precision = TP ÷ (TP + FP) = ${det.tp} ÷ ${det.tp+det.fp} = <b>${f3(det.precision)}</b>`)}${fx(`Recall = TP ÷ (TP + FN) = ${det.tp} ÷ ${n0(gt)} = <b>${f3(det.recall)}</b>`)}
    <p><b>The relation.</b> Raising the confidence threshold removes the least certain boxes. Most of those were false alarms, so precision rises; some were real damage, so recall falls. You cannot raise both by moving the threshold; you can only choose where to sit. A better model moves the whole curve up and to the right.</p>`,
    `<ul><li>For localization, <b>recall matters most</b>. A missed damage is cut out of the crop and the severity classifier never sees it. A false box only makes the union crop a little larger.</li>
    <li>That is what the error analysis found: Grad-CAM heatmaps of the whole-image model often sit on damage <i>outside</i> the YOLO boxes.</li>
    <li>The chart below uses the real CarDD validation sweep from the detector selection (${sel?sel.sweep?sel.sweep.length/3:8:8} thresholds per model). It is validation data, which is where the threshold was chosen.</li></ul>`);
  if(sel&&sel.sweep)sweepDemo(pr,sel);

  card('f1','DETECTION · ONE NUMBER','F1 score, and how it picked the threshold',
    `<p>F1 combines precision and recall into one number with the <b>harmonic mean</b>:</p>${fx(`F1 = 2 · P · R ÷ (P + R) = 2·TP ÷ (2·TP + FP + FN) = 2·${det.tp} ÷ (${2*det.tp} + ${det.fp} + ${det.fn}) = <b>${f3(det.f1)}</b>`)}
    <p>The harmonic mean punishes imbalance. A detector with precision 0.95 and recall 0.10 has an ordinary average of 0.525, but its F1 is only <b>0.18</b>: it is useless, because it finds almost nothing. F1 is high only when <i>both</i> are high. In the precision–recall plane above, the dotted curves join points of equal F1.</p>`,
    `<ul><li><b>Threshold rule (validation):</b> ${sel?sel.rule:''}</li>
    <li>YOLO11s peaks at <b>${f2(sel.confidence)}</b> with validation F1 <b>${f3(sel.models.find(m=>m.model===sel.chosen).f1)}</b>. It beat YOLO11n-100 (${f3(sel.models.find(m=>m.model==='yolo11n_e100').f1)}) by more than the required 0.01, so it was chosen.</li>
    <li>On the CarDD test the frozen choice reached F1 <b>${f3(det.f1)}</b>${sixb?`; the six-class detector, counted class-agnostically, reached ${f3(sixb.f1)}`:''}.</li>
    <li>On the severity side, F1 is computed per class and then averaged into macro-F1 (below).</li></ul>`);

  const perType=sixc&&Object.entries(sixc.per_type);
  card('ap','DETECTION · WHOLE CURVE','Average precision (AP), mAP50 and mAP50–95',
    `<p>F1 judges a single threshold. <b>Average precision</b> judges all of them: it is the <b>area under the precision–recall curve</b>, built by sweeping the threshold from high to low. Ultralytics reads precision at 101 evenly spaced recall points and averages them. AP is 1 only if every real damage is found before any false box appears.</p>${fx('AP = ∫ precision d(recall)  ·  mAP = mean of AP over classes')}
    <p><b>mAP</b> averages AP over classes. With one class (“damage”), mAP50 is simply AP at IoU 0.5. <b>mAP50–95</b> averages AP over ten IoU thresholds from 0.50 to 0.95, so it also measures how tight the boxes are.</p>`,
    `<ul><li><b>Training curves</b> (Overview tab): validation mAP50 after each epoch, which shows when training stops improving.</li>
    <li><b>Final detector, CarDD test:</b> mAP50 <b>${f3(det.mAP50)}</b>, mAP50–95 ${f3(det.mAP50_95)}${test.results.find(r=>r.role==='reference')?` (round-1 YOLO11n: ${f3(test.results.find(r=>r.role==='reference').mAP50)})`:''}.</li>
    ${sixc?`<li><b>Six-class detector.</b> Ultralytics' “mAP50” for six classes is the mean of the per-type APs (${f3(sixc.mean_over_types_mAP50)} on test). It weights easy types such as glass shatter as much as hard ones such as crack, so it cannot be compared with a one-class number. For a fair comparison both models were scored <b>class-agnostically</b>: ${f3(six.results.yolo11s_e100.mAP50)} one-class vs <b>${f3(sixc.mAP50)}</b> six-class.</li>`:''}
    ${sixv&&sixv.results.yolo11s_e100_repeat?`<li><b>Noise level.</b> Two identical YOLO11s trainings scored ${f3(sixv.results.yolo11s_e100.mAP50)} and ${f3(sixv.results.yolo11s_e100_repeat.mAP50)} validation mAP50. A difference of that size (≈${f2(Math.abs(sixv.results.yolo11s_e100_repeat.mAP50-sixv.results.yolo11s_e100.mAP50))}) can come from chance alone.</li>`:''}</ul>
    ${perType?`<p class="subhead">Test AP50 per damage type (six-class detector)</p>`:''}`);
  if(perType)group.lastChild.querySelector('.mrole').append(table(['Type','AP50','AP50–95'],perType.map(([k,v])=>[k,f3(v.AP50),f3(v.AP50_95)])));

  const rbd=det.recall_by_damages;
  card('coverage','DETECTION · PER IMAGE','Recall by number of damages per image',
    `<p>The same recall, split by how many damages an image contains. It answers a question that matters for cropping: when a car has several damages, does the detector find all of them?</p>${fx('recall(k) = found damages ÷ all damages, over images with k true boxes')}`,
    `<p>CarDD test, chosen detector:</p>${'<div class="mbars">'+Object.entries(rbd).map(([k,v])=>`<div><span>${k} damage${k==='1'?'':'s'}</span><i><b style="width:${v*100}%"></b></i><em>${pct(v)}</em></div>`).join('')+'</div>'}
    <p>Recall falls from ${pct(rbd['1'])} to ${pct(rbd['5+'])} as damages multiply. On a heavily damaged car, the crop is likely to leave some damage out. This is one reason localized severity did not beat the whole image.</p>`);

  // ===== Severity =====
  groupHead('Severity measures','The ResNet50 classifiers label each data3a image minor, moderate or severe. These measures compare their predictions with the true labels of the 245 test images. Every severity number is the mean over three training seeds.');

  const cm=card('confusion','SEVERITY · BUILDING BLOCK','Confusion matrix',
    `<p>A table of true class (rows) against predicted class (columns). The diagonal holds the correct predictions; every off-diagonal cell is a specific mistake. All the other classification measures can be read from it:</p><ul><li><b>Recall</b> of a class = its diagonal cell ÷ its <i>row</i> total.</li><li><b>Precision</b> of a class = its diagonal cell ÷ its <i>column</i> total.</li><li><b>Accuracy</b> = sum of the diagonal ÷ all images.</li></ul><p class="note">The cells are mean counts per seed (each of the three seeds predicts all ${sub.all.n} images), recomputed here from the saved predictions. Hover a cell to read it.</p>`,
    `<div id="cmdemo"></div><p>Moderate is the hard class: the whole-image model sends ${pct(ea.moderate_predicted_as.whole.minor)} of moderate images to minor and ${pct(ea.moderate_predicted_as.whole.severe)} to severe. The localized model sends ${pct(ea.moderate_predicted_as.crop.minor)} and ${pct(ea.moderate_predicted_as.crop.severe)}. Its errors lean towards “severe”.</p>`);
  if(P)confusionDemo(cm.querySelector('#cmdemo'),P);

  card('accuracy','SEVERITY · SIMPLE BUT LIMITED','Accuracy and the majority baseline',
    `<p><b>Accuracy</b> is the share of images classified correctly. It is easy to read, but with unequal classes it can look good while one class is mostly wrong.</p>${fx('Accuracy = correct predictions ÷ all predictions')}<p>The <b>majority baseline</b> is the accuracy of a “model” that always predicts the most common class. Any real model must beat it.</p>`,
    `<ul><li>Test accuracy: whole image <b>${pct(fin.variants.whole.accuracy)}</b>, localized <b>${pct(fin.variants.crop.accuracy)}</b>.</li>
    <li>Majority baseline: always “severe” = ${pct(fin.majority_accuracy)}. Both models beat it by about 20 points.</li>
    <li>Accuracy is reported but is not the main measure. The test classes are unequal (${ea.per_class.filter(r=>r.input==='whole').map(r=>`${r.support} ${r.label}`).join(', ')}), and accuracy hides that moderate recall is only ${f3(fin.variants.whole.recall['02-moderate'])}.</li></ul>`);

  const pcard=card('perclass','SEVERITY · PER CLASS','Per-class precision, recall and F1',
    `<p>The same three detection measures, computed once per severity class by treating that class as “positive” and the other two as “negative”:</p><ul><li><b>Precision(moderate):</b> of the images predicted moderate, how many really are moderate.</li><li><b>Recall(moderate):</b> of the truly moderate images, how many were predicted moderate.</li><li><b>F1(moderate):</b> their harmonic mean.</li></ul><p><b>Support</b> is the number of test images of that class.</p>`,
    `<p>Test split, mean over seeds 812–814:</p>`);
  pcard.querySelector('.mrole').append(table(['Input','Class','Support','Precision','Recall','F1'],ea.per_class.map(r=>({cls:r.label==='moderate'?'chosen':null,cells:[r.input==='crop'?'localized':'whole',r.label,String(r.support),f3(r.precision),f3(r.recall),f3(r.f1)]})),2),el('p',null,'Moderate (highlighted) is the weakest class for both inputs, at F1 ≈ 0.43. Minor images are found best. Severe images are predicted with good precision but lower recall.'));

  const wm=['minor','moderate','severe'].map(l=>pc('whole',l).f1);
  card('macro','SEVERITY · MAIN MEASURE','Macro-F1',
    `<p><b>Macro-F1</b> is the plain average of the per-class F1 scores. Each class counts equally, whatever its size. A model cannot score well by ignoring a small or difficult class.</p>${fx(`Macro-F1 = (F1<sub>minor</sub> + F1<sub>moderate</sub> + F1<sub>severe</sub>) ÷ 3 = (${wm.map(v=>v.toFixed(4)).join(' + ')}) ÷ 3 = <b>${fin.variants.whole.macro_f1.toFixed(4)}</b>`)}<p class="note">This is the whole-image model on the test split. Its alternatives are <i>weighted</i> F1, which weights by class size, and <i>micro</i> F1, which equals accuracy for single-label problems. Neither treats moderate fairly.</p>`,
    `<p>Macro-F1 is the measure used for every severity decision:</p><ul><li><b>Epoch selection:</b> each training run saves best.pt at the epoch with the highest validation macro-F1 (early stopping, patience 7).</li>
    <li><b>Crop variant:</b> ${chosenCrop?`padding ${(chosenCrop.pad*100).toFixed(0)}% was chosen with mean validation macro-F1 ${f3(chosenCrop.selection_mean)} over seeds 812–813`:''}.</li>
    <li><b>Dashboard seeds:</b> the best-validation seed per input (whole ${F.models.whole.seed}: ${f3(F.models.whole.validation_macro_f1)}, localized ${F.models.crop.seed}: ${f3(F.models.crop.validation_macro_f1)}).</li>
    <li><b>Final test:</b> whole <b>${f3(fin.variants.whole.macro_f1)}</b> vs localized <b>${f3(fin.variants.crop.macro_f1)}</b>.</li></ul>`);

  card('deltas','SEVERITY · THE RESEARCH QUESTION','ΔF1 and ΔR moderate',
    `<p>The two measures that answer the research question. Both were written into <code>final_configs.json</code> before the final test (${cfg.frozen_utc.slice(0,10)}), so they could not be chosen after seeing the results:</p>${cfg.measures.map(m=>fx(m.replace('delta_F1','ΔF1').replace('delta_R_moderate','ΔR<sub>moderate</sub>'))).join('')}<p>Above zero means localization helped; below zero means it hurt. ΔR<sub>moderate</sub> isolates the class where seeing the damage up close was expected to help most.</p>`,
    `<table class="ftable"><thead><tr><th>Comparison</th><th class="num">Estimate</th><th class="num">95% CI</th></tr></thead><tbody>
    ${r1?`<tr><td>Round 1 · ΔF1 (YOLO11n crops)</td><td class="num neg">${sgn(r1.delta_f1)}</td><td class="num">${rng(r1.delta_f1_ci)}</td></tr>`:''}
    <tr class="chosen"><td>Final · ΔF1 (YOLO11s crops)</td><td class="num">${sgn(fin.delta_f1)}</td><td class="num">${rng(fin.delta_f1_ci)}</td></tr>
    <tr class="chosen"><td>Final · ΔR moderate</td><td class="num">${sgn(fin.delta_r_moderate)}</td><td class="num">${rng(fin.delta_r_moderate_ci)}</td></tr></tbody></table>
    <p>Round 1's interval lies entirely below zero, so the old crops really hurt. Both final intervals include zero: with the stronger detector, localization neither clearly helps nor clearly hurts.</p>`);

  card('softmax','SEVERITY · PER IMAGE','Softmax probability',
    `<p>The classifier's last layer outputs three raw scores. <b>Softmax</b> turns them into three positive numbers that sum to 1, and the predicted class is the largest one (argmax).</p>${fx('p<sub>k</sub> = e<sup>z<sub>k</sub></sup> ÷ Σ<sub>j</sub> e<sup>z<sub>j</sub></sup>')}<p>Like YOLO's confidence, it ranks the classes for one image. It is not a calibrated chance of being right.</p>`,
    `<ul><li>Every reported severity measure uses only the argmax label, not the probability.</li><li>The Test model tab shows the three probabilities as bars for each input. Use them to see whether a prediction was clear-cut (one long bar) or a near tie, which is typical for moderate.</li></ul>`);

  // ===== Statistics =====
  groupHead('Statistics: how sure can we be?','A single number from one training run on 245 images is uncertain. These are the tools used to separate a real difference from noise.');

  card('seeds','STATISTICS · VARIANCE','Random seeds and run-to-run variance',
    `<p>A <b>seed</b> fixes the random choices in training: the initial weights of the new layers, the order of the images and the augmentations. Training the same model with another seed gives a slightly different result. That spread is the <b>run-to-run variance</b>.</p><p>Averaging several seeds gives a more stable estimate, and the spread shows how large a difference must be before it means anything.</p>`,
    `<ul><li>Every severity configuration was trained with <b>seeds 812, 813 and 814</b>. Reported numbers are the mean of the three.</li>
    ${chosenCrop?`<li>Chosen crop variant, validation macro-F1 by seed: ${Object.entries(chosenCrop.seeds).map(([s,v])=>`${s}: ${f3(v)}`).join(' · ')}.</li>`:''}
    ${sixv&&sixv.results.yolo11s_e100_repeat?`<li>Detector: two identical YOLO11s trainings differ by ${f2(Math.abs(sixv.results.yolo11s_e100_repeat.mAP50-sixv.results.yolo11s_e100.mAP50))} validation mAP50. The six-class gain over the better of them is smaller than that, so it is a hint, not a proven improvement.</li>`:''}</ul>`);

  const bs=card('bootstrap','STATISTICS · UNCERTAINTY','Paired bootstrap and the 95% confidence interval',
    `<p>The test set is one sample of 245 images. A different sample would give a slightly different ΔF1. The <b>bootstrap</b> estimates how much it could move, using only the data we have:</p><ol><li>Draw 245 images from the test set <i>with replacement</i>, so some images appear twice and others not at all.</li><li>Compute macro-F1 for both inputs on that draw (mean of the 3 seeds), and their difference.</li><li>Repeat 2,000 times.</li><li>The middle 95% of the 2,000 differences (2.5th to 97.5th percentile) is the <b>95% percentile interval</b>.</li></ol>
    <p><b>Paired</b> means both inputs are scored on the <i>same</i> resampled images in each draw. Easy and hard images then affect both sides equally, and the interval reflects only the difference between inputs. That makes it much narrower than comparing two separate intervals.</p>
    <p><b>Reading it.</b> If the interval excludes 0, the sign of the difference is reliable. If it includes 0, the data cannot tell which input is better, which is not the same as proving they are equal. The interval's width tells you the smallest effect this test set could detect.</p>`,
    `<p>The histogram below repeats the procedure in your browser on the saved test predictions, with a fresh random seed each time you press the button. It lands very close to the saved result (seed 812): ΔF1 ${sgn(fc.delta_F1.estimate)}, 95% CI [${sgn(fc.delta_F1.ci95_low)}, ${sgn(fc.delta_F1.ci95_high)}], ${pct(fc.delta_F1.fraction_draws_above_zero)} of draws above zero.</p>
    <ul><li>The final interval is about ±${f2((fc.delta_F1.ci95_high-fc.delta_F1.ci95_low)/2)} wide. With 245 test images, a true effect much smaller than ~0.05 macro-F1 could not be detected.</li><li>“Share of draws above zero” is how often a resample favoured cropping. It is descriptive and is not a p-value.</li></ul>`);
  if(P)bootstrapDemo(bs,P,fc);

  card('splits','STATISTICS · HONEST NUMBERS','Validation vs test, and why the test is used once',
    `<p><b>Training</b> data updates the weights. <b>Validation</b> data is used to choose things: epochs, thresholds, the detector, the crop variant. Every choice picks whatever happened to score best on validation, so validation scores end up a little too high. This is <b>selection bias</b>. The <b>test</b> split is touched only after everything is frozen, so its score is a fair estimate of performance on new images.</p>`,
    `<ul><li>Severity, whole image: validation macro-F1 <b>${f3(val.variants.whole.macro_f1)}</b> vs test <b>${f3(fin.variants.whole.macro_f1)}</b>. The validation number is the optimistic one.</li>
    <li>Detector F1: validation ${f3(sel.models.find(m=>m.model===sel.chosen).f1)}, test ${f3(det.f1)}. Here the test happened to be easier. Different splits differ in difficulty, so the gap can go either way, but only the test number is unbiased.</li>
    <li>${cfg.test_use}</li></ul>`);

  card('posthoc','STATISTICS · CAUTION','Post-hoc subsets',
    `<p>A <b>post-hoc</b> analysis is chosen after seeing the results, for example “how does cropping do when the crop is small?”. It is useful for explaining a result but not for proving one. Each subset is small, no interval was computed, and if you look at enough subsets some will look large by chance.</p>`,
    `<p>Crop − whole macro-F1 on test subsets (sizes in brackets):</p>${'<div class="mbars diverge">'+ea.subsets.filter(s=>s.subset!=='all').map(s=>{const v=s.crop_minus_whole_f1,w=Math.min(Math.abs(v)/.15,1)*50;return `<div><span>${s.subset} (${s.n})</span><i><b class="${v<0?'neg':'pos'}" style="width:${w}%;${v<0?`right:50%`:`left:50%`}"></b></i><em>${sgn(v)}</em></div>`}).join('')+'</div>'}
    <p>The pattern supports the explanation (tight crops lose context; crops that keep most of the image do not), but with 32–107 images per subset it is descriptive only.</p>`);

  const g=ea.gradcam;
  card('gradcam','EXPLANATION · WHERE THE MODEL LOOKS','Grad-CAM attention ratio',
    `<p><b>Grad-CAM</b> makes a heatmap of the image regions that pushed the classifier towards its prediction. It uses the gradients flowing into the last convolutional block (ResNet50 layer4). To turn the heatmap into one number, we compare how much of it falls inside the YOLO boxes with how much of the image the boxes cover:</p>${fx('ratio = (heatmap mass inside boxes ÷ total heatmap mass) ÷ (box area ÷ image area)')}<p>A ratio of 1 means the model pays no special attention to the detected damage; above 1 means it concentrates there. Images whose boxes cover almost nothing or almost everything are skipped, because the ratio is meaningless there.</p>`,
    `<ul><li>Whole-image models on ${g.images} detected test images: mean ratio <b>${f2(g.mean_ratio)}</b>, median ${f2(g.median_ratio)}, ${pct(g.fraction_above_1)} above 1. Per seed: ${Object.entries(g.per_seed).map(([s,v])=>`${s}: ${f2(v)}`).join(' · ')}.</li>
    <li>So the whole-image model <i>already</i> looks at the damage, and also at damage the boxes miss. A crop adds little and can remove useful evidence. The Grad-CAM panels are on the Findings tab.</li></ul>`);

  card('croparea','LOCALIZATION · INPUT SIZE','Crop area fraction, padding and fallback rate',
    `<p>The localized input is the <b>union</b> of all boxes at confidence ≥ ${f2(cfg.detector.confidence)}, enlarged by a <b>padding</b> of ${(F.models.pad*100).toFixed(0)}% of its width and height on each side and clipped to the image. The <b>crop area fraction</b> is the crop's area divided by the image's area. The <b>fallback rate</b> is the share of images with no box at all, which use the whole image.</p>${fx('crop area fraction = (crop width × crop height) ÷ (image width × image height)')}`,
    `<ul>${chosenCrop?`<li>Chosen variant on validation: fallback rate ${pct(chosenCrop.validation_fallback_rate)}, median crop area ${pct(chosenCrop.validation_median_crop_area)} of the image.</li>`:''}${round1Crop?`<li>Round 1 (YOLO11n at ${f2(round1Crop.confidence)}, padding ${(round1Crop.pad*100).toFixed(0)}%): fallback ${pct(round1Crop.validation_fallback_rate)}, median area ${pct(round1Crop.validation_median_crop_area)}.</li>`:''}
    <li>Padding was one of the variants compared on validation, because the error analysis pointed to lost context. The subsets above show the effect of crop size on the test.</li></ul>`);

  // Summary table
  groupHead('Cheat sheet','All measures at a glance.');
  const cs=el('article','mcard');cs.append(table(['Measure','Range · better','Question it answers','Used for in this project'],[
    ['IoU','0–1 · higher','Does a predicted box cover a true one?','Matching at 0.5; mAP50–95; NMS at 0.7'],
    ['Precision','0–1 · higher','Are my boxes / predictions real?','Detector threshold trade-off; per-class severity'],
    ['Recall','0–1 · higher','Did I find all the damage / all of a class?','Crop coverage; ΔR moderate'],
    ['F1','0–1 · higher','Balance of precision and recall','Choosing the detector threshold and model'],
    ['AP, mAP50','0–1 · higher','Quality across all thresholds','Detector training, test, six-class study'],
    ['mAP50–95','0–1 · higher','Are the boxes also tight?','Detector test'],
    ['Accuracy','0–1 · higher','Share correct','Reported against the majority baseline'],
    ['Macro-F1','0–1 · higher','Balanced quality over the three classes','Main severity measure, every selection'],
    ['ΔF1, ΔR moderate','−1 to 1 · > 0 favours crops','Does localization help?','Pre-registered final comparison'],
    ['95% bootstrap CI','interval','How sure is the estimate?','Excludes 0 → reliable sign'],
    ['Grad-CAM ratio','≥ 0 · 1 = neutral','Does the model look at the damage?','Explaining the final result'],
    ['Crop area / fallback','0–1','How much of the image the crop keeps','Crop variants and post-hoc subsets']].map(r=>r.map(c=>c)),4));
  group.append(cs);
}

// Interactive IoU: a fixed true box and a predicted box moved by a slider.
function iouDemo(host){
  const w=360,h=200,gtB={x:70,y:45,w:150,h:110},s=svgEl('svg',{viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':'IoU demonstration'},null);
  svgEl('rect',{x:0,y:0,width:w,height:h,rx:8,fill:'var(--bg)'},s);
  const inter=svgEl('rect',{fill:'#067568','fill-opacity':.25},s);
  svgEl('rect',{x:gtB.x,y:gtB.y,width:gtB.w,height:gtB.h,fill:'none',stroke:'#067568','stroke-width':2.5},s);
  const pb=svgEl('rect',{width:gtB.w,height:gtB.h,fill:'none',stroke:'#eb6834','stroke-width':2.5,'stroke-dasharray':'6 4'},s);
  svgEl('text',{x:gtB.x,y:gtB.y-8,'font-size':11,fill:'#067568'},s,'true box');const pl=svgEl('text',{'font-size':11,fill:'#b5452a'},s,'predicted box');
  const out=el('p','iouout'),range=document.createElement('input');range.type='range';range.min=0;range.max=150;range.value=40;range.setAttribute('aria-label','Shift the predicted box');
  const draw=()=>{const d=+range.value,p={x:gtB.x+d,y:gtB.y+d*.35,w:gtB.w,h:gtB.h},ix=Math.max(0,Math.min(gtB.x+gtB.w,p.x+p.w)-Math.max(gtB.x,p.x)),iy=Math.max(0,Math.min(gtB.y+gtB.h,p.y+p.h)-Math.max(gtB.y,p.y)),I=ix*iy,U=2*gtB.w*gtB.h-I,v=I/U;
    pb.setAttribute('x',p.x);pb.setAttribute('y',p.y);pl.setAttribute('x',p.x+p.w-72);pl.setAttribute('y',Math.min(p.y+p.h+14,h-4));
    Object.entries({x:Math.max(gtB.x,p.x),y:Math.max(gtB.y,p.y),width:ix,height:iy}).forEach(([k,val])=>inter.setAttribute(k,val));
    out.innerHTML=`IoU = ${n0(I)} ÷ ${n0(U)} = <b>${v.toFixed(2)}</b> → ${v>=.5?'<span class="ok">match: 1 TP</span>':'<span class="bad">no match: 1 FP + 1 FN</span>'} <small>(rule: IoU ≥ 0.5)</small>`};
  range.oninput=draw;draw();host.append(s,range,out);
}

// Real validation sweep: metrics vs threshold (left) and the same points in the precision–recall plane (right).
function sweepDemo(cardEl,sel){
  const models=[...new Set(sel.sweep.map(r=>r.model))],box=el('div','sweep'),ctl=el('div','sweepctl'),pick=document.createElement('select');
  models.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m+(m===sel.chosen?' (chosen)':'');pick.append(o)});pick.value=sel.chosen;pick.setAttribute('aria-label','Detector');
  const slide=document.createElement('input');slide.type='range';slide.min=0;slide.step=1;slide.setAttribute('aria-label','Confidence threshold');
  const lab=el('label',null,'Confidence threshold ');const outv=el('output');lab.append(outv,slide);ctl.append(pick,lab);
  const charts=el('div','sweepcharts'),read=el('div','sweepread'),legend=el('div','legend');
  legend.innerHTML=`<span><i class="sw" style="background:${MC.p}"></i>Precision</span><span><i class="sw" style="background:${MC.r}"></i>Recall</span><span><i class="sw" style="background:${MC.f}"></i>F1 (dashed)</span><span>◯ highest F1</span><span>right: the same thresholds as points in the precision–recall plane; dotted curves = equal F1</span>`;
  box.append(ctl,charts,legend,read);cardEl.append(box);
  let idx;
  const draw=()=>{const rows=sel.sweep.filter(r=>r.model===pick.value).sort((a,b)=>a.confidence-b.confidence);slide.max=rows.length-1;if(idx==null||idx>=rows.length)idx=rows.reduce((b,r,i)=>r.f1>rows[b].f1?i:b,0);slide.value=idx;const cur=rows[idx],best=rows.reduce((b,r)=>r.f1>b.f1?r:b);
    outv.textContent=f2(cur.confidence);charts.replaceChildren();
    // left: metric vs threshold
    const W=420,H=250,L=40,R=14,T=14,B=212,x=c=>L+(c-0)/(.55)*(W-L-R),y=v=>B-v*(B-T),s=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Precision, recall and F1 by confidence threshold'});
    for(let v=0;v<=1.001;v+=.25){svgEl('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:'#dde5e1'},s);svgEl('text',{x:L-6,y:y(v)+4,'text-anchor':'end','font-size':10,fill:'#70817e'},s,v.toFixed(2))}
    rows.forEach(r=>svgEl('text',{x:x(r.confidence),y:B+16,'text-anchor':'middle','font-size':10,fill:'#70817e'},s,r.confidence.toFixed(2)));
    svgEl('text',{x:(L+W-R)/2,y:H-4,'text-anchor':'middle','font-size':11,fill:'#70817e'},s,'confidence threshold');
    svgEl('line',{x1:x(cur.confidence),x2:x(cur.confidence),y1:T,y2:B,stroke:'#172b2a','stroke-dasharray':'3 3'},s);
    [['precision',MC.p,''],['recall',MC.r,''],['f1',MC.f,'6 4']].forEach(([k,c,dash])=>{svgEl('polyline',{points:rows.map(r=>x(r.confidence)+','+y(r[k])).join(' '),fill:'none',stroke:c,'stroke-width':2,'stroke-dasharray':dash},s);rows.forEach((r,i)=>svgEl('circle',{cx:x(r.confidence),cy:y(r[k]),r:i===idx?5:3,fill:c,stroke:'white','stroke-width':1.5},s))});
    svgEl('circle',{cx:x(best.confidence),cy:y(best.f1),r:9,fill:'none',stroke:MC.f,'stroke-width':1.5},s);
    const hit=svgEl('rect',{x:L,y:T,width:W-L-R,height:B-T,fill:'transparent',style:'cursor:pointer'},s);
    hit.onmousemove=ev=>{const pt=s.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;const p=pt.matrixTransform(s.getScreenCTM().inverse());const i=rows.reduce((b,r,j)=>Math.abs(x(r.confidence)-p.x)<Math.abs(x(rows[b].confidence)-p.x)?j:b,0);if(i!==idx){idx=i;draw()}};
    // right: precision–recall plane with iso-F1 curves
    const PW=300,PL=40,PR=14,px=v=>PL+v*(PW-PL-PR),py=v=>B-v*(B-T),q=svgEl('svg',{viewBox:`0 0 ${PW} ${H}`,role:'img','aria-label':'Precision–recall plane'});
    for(let v=0;v<=1.001;v+=.25){svgEl('line',{x1:PL,x2:PW-PR,y1:py(v),y2:py(v),stroke:'#dde5e1'},q);svgEl('text',{x:PL-6,y:py(v)+4,'text-anchor':'end','font-size':10,fill:'#70817e'},q,v.toFixed(2));svgEl('text',{x:px(v),y:B+16,'text-anchor':'middle','font-size':10,fill:'#70817e'},q,v.toFixed(2))}
    [.4,.5,.6,.7].forEach(f=>{const pts=[];for(let r=f/(2-f)+1e-3;r<=1;r+=.01){const p=f*r/(2*r-f);if(p<=1)pts.push(px(r)+','+py(p))}svgEl('polyline',{points:pts.join(' '),fill:'none',stroke:'#8a958f','stroke-dasharray':'2 3'},q);const rr=.98,pp=f*rr/(2*rr-f);svgEl('text',{x:px(rr)-2,y:py(pp)-4,'text-anchor':'end','font-size':9,fill:'#70817e'},q,'F1 '+f)});
    svgEl('text',{x:(PL+PW-PR)/2,y:H-4,'text-anchor':'middle','font-size':11,fill:'#70817e'},q,'recall');svgEl('text',{x:PL+6,y:T+16,'font-size':11,fill:'#70817e'},q,'precision ↑');
    svgEl('polyline',{points:rows.map(r=>px(r.recall)+','+py(r.precision)).join(' '),fill:'none',stroke:'#172b2a','stroke-width':1.5},q);
    rows.forEach((r,i)=>svgEl('circle',{cx:px(r.recall),cy:py(r.precision),r:i===idx?6:3.5,fill:i===idx?MC.f:'#172b2a',stroke:'white','stroke-width':1.5},q));
    charts.append(s,q);
    read.innerHTML=`<div><small>TP</small><b>${n0(cur.tp)}</b></div><div><small>FP</small><b>${n0(cur.fp)}</b></div><div><small>FN</small><b>${n0(cur.fn)}</b></div><div><small>Precision</small><b style="color:${MC.p}">${f3(cur.precision)}</b></div><div><small>Recall</small><b style="color:${MC.r}">${f3(cur.recall)}</b></div><div><small>F1</small><b style="color:${MC.f}">${f3(cur.f1)}</b></div><div><small>Images without a box</small><b>${n0(cur.no_detection_images)}</b></div>
      <p>${cur===best?`This is the highest-F1 threshold for ${pick.value}${pick.value===sel.chosen?', so it was frozen for the pipeline':''}.`:cur.confidence<best.confidence?`Lower than the best threshold (${f2(best.confidence)}): ${n0(cur.tp-best.tp)} more damages found, but ${n0(cur.fp-best.fp)} more false boxes.`:`Higher than the best threshold (${f2(best.confidence)}): ${n0(best.fp-cur.fp)} fewer false boxes, but ${n0(best.tp-cur.tp)} more damages missed.`}</p>`};
  slide.oninput=()=>{idx=+slide.value;draw()};pick.onchange=()=>{idx=null;draw()};draw();
}

// Confusion matrix from the saved test predictions, mean counts over seeds.
function confusionDemo(host,P){
  const K=P.labels.length,ctl=el('div','modes inline'),grid=el('div','cmwrap');
  ctl.innerHTML=`<label><input type="radio" name="cmv" value="whole" checked> Whole image</label><label><input type="radio" name="cmv" value="crop"> Localized crop</label>`;host.append(ctl,grid);
  const draw=v=>{const m=Array.from({length:K},()=>Array(K).fill(0));P[v].forEach(pred=>pred.forEach((p,i)=>m[P.y[i]][p]+=1/P[v].length));
    const rowT=m.map(r=>r.reduce((a,b)=>a+b,0)),colT=m[0].map((_,j)=>m.reduce((a,r)=>a+r[j],0)),t=el('table','cm');
    t.innerHTML=`<thead><tr><th></th><th colspan="${K}" class="cmaxis">predicted →</th><th></th></tr><tr><th class="cmaxis">true ↓</th>${P.labels.map(l=>`<th>${l}</th>`).join('')}<th class="num">recall</th></tr></thead>`;
    const tb=el('tbody');m.forEach((r,i)=>{const tr=el('tr');tr.append(el('th',null,P.labels[i]));r.forEach((c,j)=>{const td=el('td',i===j?'diag':null,c.toFixed(1));const f=c/rowT[i];td.style.background=`rgba(6,117,104,${(f*.85).toFixed(3)})`;if(f>.5)td.style.color='white';
      td.onmousemove=ev=>showTip(ev,[`True ${P.labels[i]}, predicted ${P.labels[j]}`,`${c.toFixed(1)} of ${rowT[i].toFixed(0)} ${P.labels[i]} images (${pct(f)} of the row)`,i===j?'Diagonal: correct':'Off-diagonal: an error',`Column total ${colT[j].toFixed(1)} → precision(${P.labels[j]}) uses it`]);td.onmouseleave=hideTip;tr.append(td)});
      tr.append(el('td','num rec',f3(r[i]/rowT[i])));tb.append(tr)});
    const pr=el('tr','prec');pr.append(el('th',null,'precision'));m.forEach((_,j)=>pr.append(el('td','num',f3(m[j][j]/colT[j]))));const acc=m.reduce((a,r,i)=>a+r[i],0)/rowT.reduce((a,b)=>a+b,0);pr.append(el('td','num',`acc ${f3(acc)}`));tb.append(pr);t.append(tb);
    grid.replaceChildren(t,el('p','note','Recall in the right column, precision in the bottom row. Precision here comes from the pooled counts; the per-class table averages the three seeds separately, so a few values differ in the third decimal.'))};
  ctl.onchange=e=>draw(e.target.value);draw('whole');
}

// Paired bootstrap of ΔF1 / ΔR moderate, recomputed in the browser from the saved predictions.
function bootstrapDemo(cardEl,P,fc){
  const y=P.y,n=y.length,K=P.labels.length,MOD=P.labels.indexOf('moderate');
  const macro=(pred,idx)=>{const tp=Array(K).fill(0),fp=Array(K).fill(0),fn=Array(K).fill(0);for(const i of idx){const t=y[i],p=pred[i];if(t===p)tp[t]++;else{fp[p]++;fn[t]++}}let s=0;for(let k=0;k<K;k++){const d=2*tp[k]+fp[k]+fn[k];s+=d?2*tp[k]/d:0}return s/K};
  const rmod=(pred,idx)=>{let a=0,b=0;for(const i of idx)if(y[i]===MOD){b++;if(pred[i]===MOD)a++}return b?a/b:0};
  const stat={f1:idx=>mean(P.crop.map(p=>macro(p,idx)))-mean(P.whole.map(p=>macro(p,idx))),rm:idx=>mean(P.crop.map(p=>rmod(p,idx)))-mean(P.whole.map(p=>rmod(p,idx)))};
  const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
  const box=el('div','boot'),ctl=el('div','sweepctl'),btn=el('button',null,'Resample 2,000 times ↻'),which=el('div','modes inline');
  which.innerHTML=`<label><input type="radio" name="bsv" value="f1" checked> ΔF1</label><label><input type="radio" name="bsv" value="rm"> ΔR moderate</label>`;ctl.append(which,btn);
  const plot=el('div','bootplot'),read=el('p','bootread');box.append(ctl,plot,read);cardEl.append(box);
  const run=()=>{const kind=box.querySelector('input[name=bsv]:checked').value,saved=kind==='f1'?fc.delta_F1:fc.delta_R_moderate;let seed=(Date.now()^0x9e3779b9)>>>0;
    const rand=()=>{seed=(seed+0x6D2B79F5)>>>0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296};
    const all=[...Array(n).keys()],point=stat[kind](all),draws=[];for(let b=0;b<2000;b++){const idx=new Array(n);for(let i=0;i<n;i++)idx[i]=Math.floor(rand()*n);draws.push(stat[kind](idx))}
    draws.sort((a,b)=>a-b);const q=p=>draws[Math.min(draws.length-1,Math.floor(p*(draws.length-1)))],lo=q(.025),hi=q(.975),above=draws.filter(d=>d>0).length/draws.length;
    const W=700,H=230,L=36,R=14,T=16,B=190,mn=kind==='f1'?-.14:-.24,mx=kind==='f1'?.11:.2,bw=kind==='f1'?.005:.01,nb=Math.round((mx-mn)/bw),cnt=Array(nb).fill(0);draws.forEach(d=>{const i=Math.floor((d-mn)/bw);if(i>=0&&i<nb)cnt[i]++});
    const top=Math.max(...cnt),x=v=>L+(v-mn)/(mx-mn)*(W-L-R),s=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Bootstrap distribution'});
    svgEl('rect',{x:x(lo),y:T,width:x(hi)-x(lo),height:B-T,fill:'#067568','fill-opacity':.08},s);
    cnt.forEach((c,i)=>{if(!c)return;const v0=mn+i*bw,inCI=v0+bw/2>=lo&&v0+bw/2<=hi;svgEl('rect',{x:x(v0)+1,y:B-c/top*(B-T-14),width:Math.max(1,x(v0+bw)-x(v0)-2),height:c/top*(B-T-14),rx:2,fill:inCI?'#067568':'#9fbaa9'},s)});
    svgEl('line',{x1:L,x2:W-R,y1:B,y2:B,stroke:'#8a958f'},s);
    for(let v=Math.ceil(mn/.05)*.05;v<=mx+1e-9;v+=.05)svgEl('text',{x:x(v),y:B+16,'text-anchor':'middle','font-size':10,fill:'#70817e'},s,Math.abs(v)<1e-9?'0':sgn(v).replace(/0+$/,'').replace(/\.$/,''));
    svgEl('line',{x1:x(0),x2:x(0),y1:T,y2:B,stroke:'#b5452a','stroke-width':1.5},s);svgEl('text',{x:x(0)+4,y:T+10,'font-size':10,fill:'#b5452a'},s,'0 = no difference');
    svgEl('line',{x1:x(point),x2:x(point),y1:T,y2:B,stroke:'#172b2a','stroke-width':2},s);svgEl('text',{x:x(point)-4,y:T+10,'text-anchor':'end','font-size':10,fill:'#172b2a','font-weight':700},s,'estimate '+sgn(point));
    svgEl('text',{x:(x(lo)+x(hi))/2,y:B+30,'text-anchor':'middle','font-size':10,fill:'#067568'},s,`95% interval [${sgn(lo)}, ${sgn(hi)}]`);
    s.setAttribute('height',H+10);s.setAttribute('viewBox',`0 0 ${W} ${H+10}`);plot.replaceChildren(s);
    read.innerHTML=`This run: estimate <b>${sgn(point)}</b>, 95% CI <b>[${sgn(lo)}, ${sgn(hi)}]</b>, ${pct(above)} of draws above zero. Saved (seed 812): ${sgn(saved.estimate)}, [${sgn(saved.ci95_low)}, ${sgn(saved.ci95_high)}], ${pct(saved.fraction_draws_above_zero)}. The dark bars are the middle 95%. ${lo<=0&&hi>=0?'The interval contains 0, so the test set cannot tell the two inputs apart.':'The interval excludes 0, so the sign is reliable.'}`};
  btn.onclick=run;which.onchange=run;run();
}
