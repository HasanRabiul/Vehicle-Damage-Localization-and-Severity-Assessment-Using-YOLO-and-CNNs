# Progress Report: Vehicle Damage Localization and Severity Assessment

ENSE 812 · Individual project · 28 October 2026

**Project:** Deep learning-based vehicle damage localization and severity assessment using YOLO and CNNs.

## Summary

- **Detector.** A YOLO11n damage detector trained on CarDD (single "damage" class, 10 epochs) reaches **mAP50 0.502** and **mAP50–95 0.313** on the 810-image validation split. A controlled re-validation reproduced the saved results.
- **Severity data.** The data3a severity dataset (minor / moderate / severe) was audited for leakage. The audit found cross-split duplicates, label conflicts and **42 images copied from CarDD**, the detector's training data. After filtering, 1,543 of 1,599 images remain.
- **Severity classifier.** A ResNet50 trained on **whole images** outperformed the same classifier trained and applied on **YOLO damage crops**. On the held-out test split, macro-F1 was **0.583 vs 0.514** (paired difference +0.069, 95% CI [+0.016, +0.121]).
- **Main finding.** Cropping to the CarDD-trained detector's boxes removes information needed to judge severity. This held even for the crop-trained classifier.

## 1. Damage detection (YOLO11n on CarDD)

**Data.** CarDD in COCO format was converted to YOLO format with all six damage types merged into one class: 2,816 training, 810 validation and 374 test images. The test split has not been used.

**Step 1: reproducing the baseline.** The saved checkpoint was re-validated without retraining.

| Metric | Saved (epoch 10) | Reproduced |
|---|---|---|
| Precision | 0.606 | 0.610 |
| Recall | 0.444 | 0.442 |
| mAP50 | 0.503 | 0.502 |
| mAP50–95 | 0.313 | 0.313 |

**Step 2: confidence-threshold audit.** The saved confusion matrix showed about 121,000 unmatched detections. This is caused by Ultralytics' default plotting threshold of 0.001, not by a model fault. At fixed thresholds (matching IoU > 0.45):

| Confidence > | False positives | Precision | Recall | Images with no detection |
|---|---|---|---|---|
| 0.001 | 120,956 | 1.4% | 96.1% | 0 |
| 0.10 | 1,736 | 37.8% | 60.4% | 3 |
| 0.25 | 391 | 66.3% | 44.0% | 88 |
| 0.50 | 74 | 85.7% | 25.5% | 338 |

**Step 3: error analysis.**
- At confidence 0.25, 265 of 810 validation images have every annotation matched with no false positives. 88 images have no detection at all.
- Visual inspection showed oversized or duplicate boxes and missed regions in images with several damaged areas.
- Some "false positives" overlap real damage but fail the one-to-one matching rule.

Figures: `outputs/experiments/yolo11n_error_analysis_20261028T043110923039Z/`.

## 2. Severity dataset audit (data3a)

The data3a dataset provides image-level severity labels in a fixed training/validation/test split. Images are small (median 259×184 px). A read-only audit (`src/audit_severity_dataset.py`) grouped byte-identical and near-identical images (dHash distance ≤ 4) and visually confirmed every flagged pair.

| Issue | Images excluded | Rule |
|---|---|---|
| Duplicates within a split | 16 | Keep one copy |
| Duplicates across splits (leakage) | 9 | Keep the training copy |
| Same image, conflicting labels | 4 | Exclude all copies |
| Validation/test images also in CarDD | 27 | Exclude, because the detector was trained on CarDD |

| Split | Minor | Moderate | Severe | Total |
|---|---|---|---|---|
| Training | 355 | 363 | 341 | 1,059 |
| Validation | 75 | 75 | 89 | 239 |
| Test | 67 | 86 | 92 | 245 |

Original files are unchanged; a manifest records which images are used. Most CarDD copies were in the test split's minor class, so removing them made the test split somewhat imbalanced.

## 3. Severity classification: whole image vs YOLO crop

**Design.** Two input variants were compared with the same classifier and training recipe. All settings were fixed before training.

- **Whole image:** the full data3a photograph.
- **YOLO crop:** the union of all detections above confidence 0.25, padded by 10% on each side. If there was no detection, the whole image was used. This fallback applied to 49% of training, 50% of validation and 38% of test images; those images were kept, not discarded.

**Training.**
- ResNet50 with ImageNet weights and a new 3-class output layer.
- 5 epochs with only the new output layer trained, then up to 25 epochs of full fine-tuning (AdamW, cosine schedule, label smoothing 0.1).
- Light augmentation; early stopping on validation macro-F1.
- Three seeds (812–814) per variant.
- The test split was evaluated **once**, after both variants and all checkpoints were fixed.

**Results** (mean over 3 seeds; 95% percentile bootstrap CI over images, 2,000 resamples):

| | Whole image | YOLO crop |
|---|---|---|
| Validation macro-F1 | 0.683 ± 0.002 | 0.646 ± 0.007 |
| **Test macro-F1** | **0.583** [0.533, 0.631] | 0.514 [0.465, 0.562] |
| Test accuracy | 57.7% | 51.0% |
| Test ordinal MAE (0–2) | 0.448 | 0.531 |
| Test recall: minor / moderate / severe | 0.81 / 0.43 / 0.55 | 0.66 / 0.33 / 0.57 |
| Test minor↔severe errors (per seed) | 0.0 minor→severe, 6.0 severe→minor | 3.7 / 6.3 |

Always predicting the most common class gives 37.6% test accuracy. The paired test difference in macro-F1 (whole minus crop) is +0.069, 95% CI [+0.016, +0.121].

**Cross-input check** (test macro-F1):

| | Evaluated on whole | Evaluated on crop |
|---|---|---|
| Trained on whole | 0.583 | 0.528 |
| Trained on crop | 0.570 | 0.514 |

Cropping at inference costs about 0.05–0.06 macro-F1 regardless of training input.

**Why crops hurt.** Visual inspection of the crops showed several patterns. The detector often boxes a wheel, headlight or panel corner rather than the main damage, and it often finds nothing on heavily wrecked cars. Some crops are only 50–70 px wide. Severity depends on the extent of damage, and a tight crop removes that context. The detector was trained on CarDD close-ups of localized damage, which differ from data3a's whole-car crash photos.

## 4. Limitations

- **Small, noisy data.** 1,059 low-resolution training images with subjective labels; the two conflicting-label pairs and visually questionable labels suggest a ceiling near 0.7 macro-F1.
- **Validation is optimistic.** Validation results are higher than test (0.683 vs 0.583) because epochs were selected on validation and the splits differ. The test confidence intervals are about ±0.05.
- **Moderate is the weakest class.** Most errors are between adjacent levels.
- **Fine-tuning added little.** For two of three whole-image seeds, the best epoch came before the backbone was unfrozen.
- **Scope of the test result.** It applies to the filtered test split. The 32 excluded test images (duplicates, conflicts and CarDD copies) were not evaluated.
- **The detector is still a preliminary 10-epoch model.**

## 5. Current system and next steps

The local DamageLab dashboard (`webapp/`) now runs the YOLO detector and the whole-image severity classifier on uploaded images. The classifier is the seed-813 checkpoint in `models/resnet50/`, the best seed on validation. The dashboard also shows the saved whole-vs-crop comparison.

Possible next steps:

1. **Train the detector longer**, or on CarDD's six damage types, and evaluate it once on the CarDD test split.
2. **Use the detector only for localization**, and look at crop strategies with more context (larger padding, minimum crop size) as a validation-only study.
3. **Improve the severity classifier:** a frozen-backbone baseline and stronger regularisation, given the overfitting observed.
4. **Write the final report** with the figures listed below.

## Artifacts

All under `outputs/experiments/`:

| Step | Folder |
|---|---|
| Baseline reproduction | `yolo11n_baseline_val_20261028T041058565345Z/` |
| Threshold audit | `yolo11n_threshold_audit_20261028T041935763377Z/` |
| Detector error analysis | `yolo11n_error_analysis_20261028T043110923039Z/` |
| Severity data audit | `severity_data3a_audit_20261028T055254716325Z/` (manifest, duplicate and CarDD-overlap panels) |
| YOLO crops | `severity_crops_20261028T060434694326Z/` |
| Training runs | `severity_resnet50_{whole,crop}_seed{812,813,814}_*/` |
| Validation evaluation | `severity_eval_validation_20261028T064807587408Z/` |
| Test evaluation | `severity_eval_test_20261028T064849642083Z/` (cross-input figure, confusion matrices, error panels) |

Scripts are in `src/`.
