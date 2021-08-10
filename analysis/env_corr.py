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
import numpy as np
import pandas as pd
import mne
from mne.minimum_norm import apply_inverse_epochs
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
corr_dir = os.path.join(results_dir, 'envelope-correlations')
for _dir in (corr_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
inverse_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
orientation_constraint = (
    '' if inverse_params['orientation_constraint'] == 'loose' else
    f"-{inverse_params['orientation_constraint']}")

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)

# load ROIs
rois = load_params(os.path.join(param_dir, 'rois.yaml'))
roi_regexp = '|'.join(rois)

# find out how many events we can realistically keep
n_good_epochs = dict(attend=dict(), ignore=dict())
for subj in subjects:
    _dir = os.path.join(data_root, subj, 'epochs')
    fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
    epochs = mne.read_epochs(os.path.join(_dir, fname), preload=False)
    n_good = dict(Counter(epochs.events[:, -1]))
    for cond, _id in epochs.event_id.items():
        n_good_epochs[cond][subj] = n_good[_id]

df = pd.DataFrame.from_dict(n_good_epochs)
n_good_overall = df.min()  # XXX TODO

for subj in subjects:
    # load epochs
    _dir = os.path.join(data_root, subj, 'epochs')
    fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
    epochs = mne.read_epochs(os.path.join(_dir, fname), preload=False)
    # load inverse
    inverse_fname = f'{subj}-{lp_cut}-sss-meg{orientation_constraint}-inv.fif'
    inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_fname)
    # apply inverse
    snr = 1.0  # assume lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = inverse_params['method']
    pick_ori = inverse_params['estimate_type']
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                pick_ori=pick_ori, return_generator=True)
    # get 'aparc' labels
    # (don't pass experiment-specific subjects_dir, use system-default one)
    labels = mne.read_labels_from_annot(
        'sample', parc='aparc', subjects_dir=None, regexp=roi_regexp)
    label_colors = [label.color for label in labels]
    # get average signal in each label (mean_flip reduces signal cancellation)
    src = inverse_operator['src']
    label_timeseries = mne.extract_label_time_course(
        stcs, labels, src, mode='mean_flip', return_generator=True)
    # compute envelope correlation
    corr = mne.connectivity.envelope_correlation(label_timeseries)
    # save results
    fname = f'{subj}-theta-band-envelope-correlation.npy'
    np.save(os.path.join(corr_dir, fname), corr)


    # XXX TODO RESUME HERE
    fig = plt.figure(figsize=(6, 6))
# let's plot this matrix
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
fig.tight_layout()
del epochs, stcs, label_ts