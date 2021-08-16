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
import matplotlib.pyplot as plt
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

# colors
label_colors = {
    'caudalanteriorcingulate': '#D1BBD7',  # purple, light
    'inferiorparietal': '#4EB265',  # green, dark
    'inferiortemporal': '#AE76A3',  # purple
    'insula': '#CAE0AB',  # green, light
    'lateralorbitofrontal': '#882E72',  # purple, dark
    'medialorbitofrontal': '#F7F056',  # yellow
    'middletemporal': '#1965B0',  # blue, dark
    'posteriorcingulate': '#F4A736',  # orange, light
    'rostralanteriorcingulate': '#5289C7',  # blue
    'superiorparietal': '#E8601C',  # orange
    'superiortemporal': '#7BAFDE',  # blue, light
    'temporalpole': '#DC050C',  # red
    'inferiorfrontal': '#90C987',  # green pars{opercular|orbital|triangular}is
}

# '#D1BBD7',  # purple, light
# '#AE76A3',  # purple
# '#882E72',  # purple, dark
# '#1965B0',  # blue, dark
# '#5289C7',  # blue
# '#7BAFDE',  # blue, light
# '#4EB265',  # green, dark
# '#90C987',  # green
# '#CAE0AB',  # green, light
# '#F7F056',  # yellow
# '#F4A736',  # orange, light
# '#E8601C',  # orange
# '#DC050C',  # red

# load ROIs
rois = load_params(os.path.join(param_dir, 'rois.yaml'))
rois_to_merge = ('parsorbitalis', 'parsopercularis', 'parstriangularis')
rois = set(rois) - set(rois_to_merge)
roi_regexp = '|'.join(rois)
# get regular labels
labels = mne.read_labels_from_annot(
    'ANTS6-0Months3T', parc='aparc', subjects_dir=subjects_dir,
    regexp=roi_regexp)
# get merged labels
for h in ('lh', 'rh'):
    _regexp = '|'.join((f'{roi}-{h}' for roi in rois_to_merge))
    _labels = mne.read_labels_from_annot(
        'ANTS6-0Months3T', parc='aparc', subjects_dir=subjects_dir,
        regexp=_regexp)
    merged_label = sum(_labels[1:], _labels[0])
    labels.append(merged_label)
# set label colors
for label in labels:
    if '+' in label.name:  # merged label
        label.color = label_colors['inferiorfrontal']
    else:
        label.color = label_colors[label.name.rsplit('-', maxsplit=1)[0]]


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
    ax = fig.add_subplot()
    ax.imshow(corr, cmap='magma', clim=np.percentile(corr, [5, 95]))
