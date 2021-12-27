#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Envelope correlation of face-to-face data, in several bands.

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_connectivity import envelope_correlation
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params, get_skip_regexp

# flags
n_sec = 7  # epoch duration to use

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
param_dir = os.path.join('..', 'params')
epo_dir = os.path.join(results_root_dir, 'epochs')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
for _dir in (conn_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
inv_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
orientation_constraint = (
    '' if inv_params['orientation_constraint'] == 'loose' else
    f"-{inv_params['orientation_constraint']}")
cov_type = inv_params['cov_type']  # 'erm' or 'baseline'

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)
freq_bands = load_params(os.path.join(param_dir, 'freq_bands.yaml'))
analysis_bands = load_params(os.path.join(param_dir, 'analysis_bands.yaml'))
labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))

custom_parcellations = ('hickok', 'corbetta', 'friederici', 'f2f')

for subj in subjects:
    # check if we should skip
    if subj in excludes and n_sec in excludes[subj]:
        continue
    # load inverse
    inv_fnames = dict(
        erm=f'{subj}-meg-erm{orientation_constraint}-inv.fif',
        baseline=f'{subj}-{lp_cut}-sss-meg{orientation_constraint}-inv.fif'
    )
    inv_fname = inv_fnames[cov_type]
    inv_fpath = os.path.join(data_root, subj, 'inverse', inv_fname)
    inv_operator = read_inverse_operator(inv_fpath)
    src = inv_operator['src']
    snr = 1.0  # assume low SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = inv_params['method']
    pick_ori = (None if inv_params['estimate_type'] == 'magnitude' else
                inv_params['estimate_type'])
    # load epochs
    for freq_band in analysis_bands:
        slug = f'{subj}-{n_sec}sec'
        epo_fname = f'{subj}-{freq_band}-band-filtered-{n_sec}sec-epo.fif'
        epochs = mne.read_epochs(os.path.join(epo_dir, epo_fname))
        # do hilbert in sensor space (faster)
        epochs.apply_hilbert()
        # apply inverse
        stcs = apply_inverse_epochs(epochs, inv_operator, lambda2, method,
                                    pick_ori=pick_ori)
        # get average signal in each label
        for parcellation, skips in labels_to_skip.items():
            # load labels
            prefix = (parcellation if parcellation in
                      custom_parcellations else '')
            regexp = get_skip_regexp(skips, prefix=prefix)
            labels = mne.read_labels_from_annot(
                subj, parcellation, regexp=regexp,
                subjects_dir=subjects_dir)
            # strip off the model name
            if parcellation in custom_parcellations:
                label_names = [label.name.split('-', maxsplit=1)[1]
                               for label in labels]
            else:
                label_names = [label.name for label in labels]
            assert len(label_names) == len(set(label_names))
            # mode=mean doesn't risk signal cancellation if using only the
            # magnitude of a (usu. free orientation constraint) inverse
            mode = ('mean' if inv_params['estimate_type'] == 'magnitude'
                    else 'auto')
            label_timeseries = mne.extract_label_time_course(
                stcs, labels, src, mode=mode, return_generator=False)
            label_timeseries = np.array(label_timeseries)
            # compute connectivity across all trials & in each condition
            for condition in tuple(epochs.event_id) + ('allconds',):
                _ids = (tuple(epochs.event_id.values())
                        if condition == 'allconds' else
                        (epochs.event_id[condition],))
                indices = np.in1d(epochs.events[:, -1], _ids)
                conn = envelope_correlation(label_timeseries[indices],
                                            names=label_names
                                            ).combine('median')
                # save
                conn_fname = (f'{parcellation}-{subj}-{condition}-'
                              f'{freq_band}-band-{n_sec}sec-'
                              'envelope-correlation.nc')
                conn_fpath = os.path.join(conn_dir, conn_fname)
                conn.save(conn_fpath)
