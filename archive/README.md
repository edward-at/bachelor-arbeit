# Archive

Diagnostics that were used during the analysis and are not part of the pipeline. No thesis
figure, table or number depends on them.

| Script | Purpose | To run |
|---|---|---|
| `regime_heatmap_sweep.py` | shows the regime profile heatmap for every k of the sweep in an interactive window, writes no file | `PYTHONPATH=05_regime_discovery python archive/regime_heatmap_sweep.py` |
| `inspect_clip_mapping.py` | checks how `clip_id` in `std_data` maps to the clip files in `inertial_data`, writes no file | `python archive/inspect_clip_mapping.py` |
| `curation_info_dump.py` | read-only console summary of manifests, run metadata, the tier-threshold check, the coverage numbers, and a search for the literature PDFs behind the 0.6 g and 0.75 g thresholds | `PYTHONPATH=07_subset_curation:04_exploratory_analysis python archive/curation_info_dump.py` |
