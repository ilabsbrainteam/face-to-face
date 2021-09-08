#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Envelope correlation of face-to-face data (for connectivity analysis). The data
are expected to be already band-pass filtered (theta band) and epoched.

authors: Daniel McCloy
license: MIT
"""

import os
from collections import Counter
import pandas as pd
import mne
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params

# config paths
data_root, *_ = load_paths()

# load other config values
subjects = load_subjects()
mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)
min_epochs = load_params(os.path.join('..', 'params', 'min_epochs.yaml'))

# find out how many events we can realistically keep
n_good_epochs = dict(attend=dict(), ignore=dict())
for subj in subjects:
    _dir = os.path.join(data_root, subj, 'epochs')
    fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
    epochs = mne.read_epochs(os.path.join(_dir, fname), preload=False,
                             verbose=False)
    n_good = dict(Counter(epochs.events[:, -1]))
    for cond, _id in epochs.event_id.items():
        n_good_epochs[cond][subj] = n_good[_id]

df = pd.DataFrame.from_dict(n_good_epochs)
# sort by smallest value in either column
df = (df.assign(min=df.values.min(axis=1), max=df.values.max(axis=1))
        .sort_values(['min', 'max'])
        .drop(columns=['min', 'max']))
print()
print(df)
print()
print("THESE SUBJECTS DON'T HAVE ENOUGH EPOCHS; ADD THEM TO "
      "../params/skip_subjects.yaml:")
print()
print(df.query(f'attend < {min_epochs} or ignore < {min_epochs}'))
print()
