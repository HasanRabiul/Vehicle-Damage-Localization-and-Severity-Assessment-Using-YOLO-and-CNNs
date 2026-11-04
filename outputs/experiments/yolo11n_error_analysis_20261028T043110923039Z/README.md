# Validation error-analysis figures

Fixed confidence > 0.25 and matching IoU > 0.45, used for inspection only; no operational threshold is selected. Predictions are frozen post-NMS outputs from step 1. Rounded export coordinates/scores limit exact equivalence to in-memory evaluation.

Each figure shows four reproducibly sampled validation images, with ground truth on the left and predictions on the right. Blue: matched annotation; magenta: missed annotation; green: matched prediction; orange: unmatched prediction. Matching all damage annotations does not imply a correct severity classification. Categories overlap and these illustrative samples do not estimate error prevalence.

No-detection cases must remain in the future severity evaluation through a predefined fallback rather than being discarded. No training, new inference, or test-set evaluation was performed. Original images, labels and checkpoint remain unchanged.

selection.json records pool sizes, filenames, seed and image hashes. per_image.csv records counts for all 810 validation images. Counts agree with the step-2 audit.

## Qualitative inspection

The sampled false-positive panels show oversized and duplicate predictions around damaged regions, as well as predictions that fail to match individual annotations. No-detection panels include visible scratches, dents, and bumper damage. These are observations from illustrative sampled panels, not a systematic taxonomy or prevalence estimate. An unmatched prediction may overlap real damage but fail the matching rule, so FP does not always mean an undamaged region was detected.
