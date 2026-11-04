# Validation visual error analysis

Confidence > 0.25; matching IoU > 0.45. Frozen predictions, no fresh inference or training. Test split unused.

{
  "totals": {
    "tp": 768,
    "fp": 391,
    "fn": 976
  },
  "eligible_counts": {
    "correct_localization": 265,
    "false_positives": 295,
    "missed_damage": 380,
    "no_detections": 88
  }
}

Examples are deterministically selected and are not random prevalence estimates. See selection.json for selection rules and per_image_matches.json for coordinates and match indices. Categories overlap. A missed box may correspond to a partially localized region below the matching IoU. Unmatched predictions are not necessarily undamaged regions: duplicates and poor localization also count as false positives.

No severity labels are inferred and no crop threshold is selected. No-detection examples must be retained in later severity comparisons through an explicit fallback policy. Export rounding is a limitation.
