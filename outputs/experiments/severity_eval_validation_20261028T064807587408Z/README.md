# Severity classifier validation evaluation

All six best.pt checkpoints (whole/crop x seeds 812-814) evaluated on the 239 filtered validation images with each input type. Diagonal cells (trained and evaluated on the same input) are the planned comparison; off-diagonal cells measure input mismatch. detected/fallback subsets split by whether YOLO found damage. summary.json bootstrap gives paired 95% CIs (2,000 image resamples) for the seed-mean of the two planned configurations and their difference.

Validation was also used for epoch selection, so these numbers are optimistic. No test image was loaded. Error panels are illustrative random samples (seed 812), not prevalence estimates.
