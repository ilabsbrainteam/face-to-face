#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Envelope correlation of face-to-face data (for connectivity analysis). The data
are expected to be already band-pass filtered (theta band) and epoched.

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_connectivity import envelope_correlation, spectral_connectivity
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params, get_roi_labels

# flags
cov_type = 'erm'  # 'erm' or 'baseline'
conn_type = 'wpli'  # 'wpli' or 'env'

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
conn_dirname = 'envelope-correlations' if conn_type == 'env' else 'wpli'
conn_fname = ('theta-band-envelope-correlation' if conn_type == 'env' else
              'wpli')
conn_dir = os.path.join(results_dir, conn_dirname)
for _dir in (conn_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
n_epochs = load_params(os.path.join('..', 'params', 'min_epochs.yaml'))
inv_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
orientation_constraint = (
    '' if inv_params['orientation_constraint'] == 'loose' else
    f"-{inv_params['orientation_constraint']}")

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)

for subj in subjects:
    # load labels
    labels = get_roi_labels(subj, param_dir)
    label_names = [lab.name for lab in labels]
    # load epochs
    _dir = os.path.join(data_root, subj)
    epo_fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
    epochs = mne.read_epochs(os.path.join(_dir, 'epochs', epo_fname),
                             preload=False)

    # truncate to same number of epochs per subject
    new_eve = np.empty((0, 3), dtype=int)
    event_codes = list(epochs.event_id.values())
    for code in event_codes:
        mask = epochs.events[:, -1] == code
        new_eve = np.vstack((new_eve, epochs.events[mask][:n_epochs]))
    new_eve = new_eve[np.argsort(new_eve[:, 0])]
    selection = np.nonzero(np.in1d(epochs.events[:, 0], new_eve[:, 0]))[0]
    epochs = epochs[selection]
    # load inverse
    inv_fnames = dict(
        erm=f'{subj}-meg-erm{orientation_constraint}-inv.fif',
        basline=f'{subj}-{lp_cut}-sss-meg{orientation_constraint}-inv.fif'
    )
    inv_fname = inv_fnames[cov_type]
    inv_fpath = os.path.join(_dir, 'inverse', inv_fname)
    inv_operator = read_inverse_operator(inv_fpath)
    src = inv_operator['src']
    snr = 1.0  # assume lower SNR for single epochs
    lambda2 = 1.0 / snr ** 2
    method = inv_params['method']
    pick_ori = (None if inv_params['estimate_type'] == 'magnitude' else
                inv_params['estimate_type'])
    # loop over conditions
    for condition in epochs.event_id:
        this_epochs = epochs[condition]
        # apply inverse
        stcs = apply_inverse_epochs(this_epochs, inv_operator, lambda2,
                                    method, pick_ori=pick_ori,
                                    return_generator=True)
        # get avg signal in each label (mean_flip reduces signal cancellation)
        label_timeseries = mne.extract_label_time_course(
            stcs, labels, src, mode='mean_flip', return_generator=True)
        # compute connectivity
        if conn_type == 'env':
            conn = envelope_correlation(label_timeseries,
                                        names=label_names).combine('mean')
        else:
            # XXX bug in saving the object when n_jobs > 1
            # https://github.com/mne-tools/mne-connectivity/issues/40
            conn = spectral_connectivity(
                label_timeseries, names=label_names,
                method='wpli2_debiased', sfreq=epochs.info['sfreq'],
                mode='multitaper', fmin=(0, 4, 8), fmax=(4, 8, 12),
                n_jobs=1)
        # save
        conn_fpath = os.path.join(conn_dir,
                                  f'{subj}-{condition}-{conn_fname}.nc')
        conn.save(conn_fpath)
