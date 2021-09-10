#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Envelope correlation of face-to-face data, in several bands.

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_connectivity import envelope_correlation
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params

# flags
cov_type = 'erm'  # 'erm' or 'baseline'

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
epo_dir = os.path.join(results_dir, 'epochs')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
for _dir in (conn_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
inv_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
orientation_constraint = (
    '' if inv_params['orientation_constraint'] == 'loose' else
    f"-{inv_params['orientation_constraint']}")

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)

freq_bands = ('delta', 'theta', 'beta')

for subj in subjects:
    # load labels
    labels = mne.read_labels_from_annot(subj, 'aparc',
                                        subjects_dir=subjects_dir)
    label_names = [label.name for label in labels]
    # load inverse
    inv_fnames = dict(
        erm=f'{subj}-meg-erm{orientation_constraint}-inv.fif',
        baseline=f'{subj}-{lp_cut}-sss-meg{orientation_constraint}-inv.fif'
    )
    inv_fname = inv_fnames[cov_type]
    inv_fpath = os.path.join(data_root, subj, 'inverse', inv_fname)
    inv_operator = read_inverse_operator(inv_fpath)
    src = inv_operator['src']
    snr = 1.0  # assume lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = inv_params['method']
    pick_ori = (None if inv_params['estimate_type'] == 'magnitude' else
                inv_params['estimate_type'])
    # load epochs
    for freq_band in freq_bands:
        epo_fname = f'{subj}-{freq_band}-band-filtered-epo.fif'
        epochs = mne.read_epochs(os.path.join(epo_dir, epo_fname))
        # loop over conditions
        for condition in epochs.event_id:
            this_epochs = epochs[condition]
            # apply inverse
            stcs = apply_inverse_epochs(this_epochs, inv_operator, lambda2,
                                        method, pick_ori=pick_ori,
                                        return_generator=True)
            # get average signal in each label
            # (mean_flip reduces signal cancellation)
            label_timeseries = mne.extract_label_time_course(
                stcs, labels, src, mode='mean_flip', return_generator=True)
            # compute connectivity
            conn = envelope_correlation(label_timeseries,
                                        names=label_names).combine('mean')
            # save
            conn_fname = (f'{subj}-{condition}-{freq_band}-band'
                          '-envelope-correlation.nc')
            conn_fpath = os.path.join(conn_dir, conn_fname)
            conn.save(conn_fpath)
