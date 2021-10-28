# Preprocessing

1. `morph_labels.py` (converts labels from fsaverage to template brain)
2. `run_coreg.py` (auto-aligns & warps template brains to match subject sensor
    locations, and opens each alignment for visual quality check)
3. `run_mnefun.py` (runs the automated preprocessing pipeline to create source
   spaces, inverses, etc)
4. `do_autoreject.py` (epochs at a few different lengths/overlaps and stores
    autoreject thresholds for each)
5. **MANUAL STEP** check resulting epoch counts, and determine what cutoff is 
   appropriate for each epoch duration. Enter those values as `n_min` in
   `../params/min_epochs.yaml`.
6. `filter_and_epoch.py` (downsamples, filters to various freq bands, epochs w/
   reject values determined in step 4, and equalizes event counts across
   conditions and subjects)

`f2f_score.py` should not be run directly, it is called from `run_mnefun.py`
