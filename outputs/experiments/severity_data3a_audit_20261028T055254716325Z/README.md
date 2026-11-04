# data3a severity dataset audit

Read-only audit of data/data3a (3 classes: 01-minor, 02-moderate, 03-severe). Original files are not moved or edited; manifest.csv marks which images to use.

Duplicates are byte-identical (SHA-256) or near-identical (dHash Hamming distance <= 4) images, grouped transitively. Groups with conflicting labels are excluded entirely. Otherwise one copy is kept with priority training > validation > test, so leaked validation/test copies and same-split duplicates are excluded.

Near-duplicate matching at these small resolutions (median about 259x184) can produce false positives and misses; cross_split_duplicates.png and label_conflicts.png show every pair for visual confirmation. CarDD near-matches were visually confirmed as the same photographs. Because the YOLO detector was trained on CarDD, validation/test images matching CarDD are excluded (reason cardd_overlap); training copies are kept and flagged in the cardd_near column.

No training or model evaluation was performed.
