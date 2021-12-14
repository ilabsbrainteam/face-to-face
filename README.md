# Face-to-face connectivity analysis

0. check that all files in `params/` have the values you want
1. check the preprocessing params in `preprocessing/mnefun_params.yaml`
2. run `preprocessing/run_coreg.py` (run once) to generate copies of the
   surrogate MRI scaled to each subject's headshape points / cardinal
   landmarks. Also generates the `-trans.fif` file
3. run `preprocessing/run_mnefun.py`
4. run `preprocessing/morph_labels.py`
5. run `analysis/*.py` (see `analysis/README.md`)
