# Parcellations and labels

1. `morph_labels.py` (converts labels from fsaverage to template brain)

2. **MANUAL STEP** run `define_rois_helper.py` interactively to help
   distinguish the subdivisions in `aparc_sub` and choose which ones to include
   in the various ROIs defined in `../params/rois.yaml`

3. `create_custom_parcellation.py` (uses `../params/rois.yaml` to create a
   parcellation where each ROI is its own label, and the remaining labels are
   either `aparc` labels or whatever is left over in an `aparc` label after
   removing areas that are part of our custom ROIs)

4. `copy_custom_parcellation.py` (copy the custom parcellation from our
   surrogate brain to each subject)

# Preprocessing

1. `run_coreg.py` (auto-aligns & warps template brains to match subject sensor
    locations, and opens each alignment for visual quality check)

    a. `rescale_labels.py` (generally not needed; a run-once file to re-copy
       labels from surrogate to subjects, because the originally-copied labels
       were corrupted)

2. `run_mnefun.py` (runs the automated preprocessing pipeline to create source
   spaces, inverses, etc)

3. `do_autoreject.py` (epochs at a few different lengths/overlaps and stores
    autoreject thresholds for each)

4. **MANUAL STEP** check resulting epoch counts, and determine what cutoff is 
   appropriate for each epoch duration. Enter those values as `n_min` in
   `../params/min_epochs.yaml`.

5. `filter_and_epoch.py` (downsamples, filters to various freq bands, epochs w/
   reject values determined in step 4, and equalizes event counts across
   conditions and subjects)

`f2f_score.py` should not be run directly, it is called from `run_mnefun.py`
