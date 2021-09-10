#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Epoching of face-to-face data, suitable for connectivity analysis.

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
import mne
from mne.externals.h5io import read_hdf5
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params

# flags
new_sfreq = 1000

# config paths
data_root, subjects_dir, results_dir = load_paths()
epo_dir = os.path.join(results_dir, 'epochs')
for _dir in (epo_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
n_epochs = load_params(os.path.join('..', 'params', 'min_epochs.yaml'))
mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)

freq_bands = dict(delta=(0, 4),
                  theta=(4, 8),
                  beta=(12, 30))  # for sanity checks

for subj in subjects:
    # load raw and events
    _dir = os.path.join(data_root, subj)
    raw_fname = f'{subj}_allclean_fil{lp_cut}_raw_sss.fif'
    raw_fpath = os.path.join(_dir, 'sss_pca_fif', raw_fname)
    eve_fpath = os.path.join(_dir, 'lists', f'ALL_{subj}-eve.lst')
    raw = mne.io.read_raw_fif(raw_fpath)
    events = mne.read_events(eve_fpath)
    # get the event dict from the auto-epoching done by mnefun
    epo_fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
    event_id = mne.read_epochs(os.path.join(_dir, 'epochs', epo_fname),
                               preload=False, proj=False, verbose=False
                               ).event_id
    # downsample
    raw, events = raw.load_data().resample(new_sfreq, events=events)
    # filter
    for freq_band, (l_freq, h_freq) in freq_bands.items():
        filtered_raw = raw.copy().filter(l_freq, h_freq, n_jobs='cuda')
        # load autoreject params computed by mnefun
        reject_fname = f'All_{lp_cut}-sss_{subj}-reject.h5'
        reject = read_hdf5(os.path.join(_dir, 'epochs', reject_fname))
        # epoch
        epochs = mne.Epochs(filtered_raw, events=events, event_id=event_id,
                            reject=reject, baseline=None, preload=True)
        del filtered_raw
        # truncate to same number of epochs per subject
        new_eve = np.empty((0, 3), dtype=int)
        event_codes = list(epochs.event_id.values())
        for code in event_codes:
            mask = epochs.events[:, -1] == code
            new_eve = np.vstack((new_eve, epochs.events[mask][:n_epochs]))
        new_eve = new_eve[np.argsort(new_eve[:, 0])]
        selection = np.nonzero(np.in1d(epochs.events[:, 0], new_eve[:, 0]))[0]
        epochs = epochs[selection]
        epo_fname = f'{subj}-{freq_band}-band-filtered-epo.fif'
        epochs.save(os.path.join(epo_dir, epo_fname), overwrite=True)
