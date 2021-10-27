# Preprocessing

1. `run_coreg.py`
2. `run_mnefun.py`
3. `do_autoreject.py` (epochs at a few different lengths/overlaps and stores
    autoreject thresholds for each)
4. check resulting epoch counts, and determine what cutoff is appropriate for
   each epoch duration. Enter those values as `n_min` in
   `../params/min_epochs.yaml`.
5. `filter_and_epoch.py` (downsamples, filters to various freq bands, epochs w/
   reject values determined in step 3, and equalizes event counts across
   conditions and subjects)

`f2f_score.py` should not be run directly, it is called from `run_mnefun.py`
