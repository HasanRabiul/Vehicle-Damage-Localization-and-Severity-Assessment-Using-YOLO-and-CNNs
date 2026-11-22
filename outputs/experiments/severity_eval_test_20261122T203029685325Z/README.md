# Severity classifier test evaluation

All six best.pt checkpoints (whole/crop x seeds 812-814) evaluated on the 245 filtered test images with each input type. Diagonal cells (trained and evaluated on the same input) are the planned comparison; off-diagonal cells measure input mismatch. detected/fallback subsets split by whether YOLO found damage. summary.json bootstrap gives paired 95% CIs (2,000 image resamples) for the seed-mean of the two planned configurations and their difference.

Single pre-approved test evaluation; configurations and checkpoints were fixed on validation beforehand and nothing is re-tuned from these results. Error panels are illustrative random samples (seed 812), not prevalence estimates.
