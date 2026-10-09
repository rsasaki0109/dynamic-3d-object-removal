# Changelog

## 0.6.0

- Clean accumulated maps from a pose-aware multi-scan manifest with the `fusion`
  and `range_scan_ratio` CLI workflows. Save cleaned points, boolean keep masks
  and JSON summaries of effective settings. Choose explicit short-window or
  long-map fusion presets.
- Require explicit deskew metadata for those manifests. Experimental undeskewed
  inputs require an opt-in and retain that status in the summary; rigid poses
  do not deskew sweeps.
- Add optional visibility-gated temporal filtering and Windows parallel fusion.
- Provide a download-free first-map guide and helpers for preparing maps from
  scans and checking saved outputs against masks.
- Correct AV2 benchmark ground protection: apply the source cutoff in ego
  coordinates before pose alignment, without reusing it as absolute map Z.
  Reference-based validation preserves historical settings.
- Add reproducible API/CLI validation, multi-scene records, and diagnostic
  experiments. Neighbor-column inference remains research-only and does not
  change production defaults. Results depend on sensor density and scene;
  removal counts alone do not measure accuracy.

NumPy remains the only required dependency. Optional dataset and plotting tools
are separate from the core library. Public demo behavior is unchanged by the
recent diagnostics and onboarding helpers.
