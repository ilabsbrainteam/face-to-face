#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Epoching of face-to-face data, suitable for connectivity analysis.

authors: Daniel McCloy
license: MIT
"""

import os
import yaml
from collections import defaultdict
import numpy as np
import mne
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params

# config paths
data_root, subjects_dir, results_dir = load_paths(include_inv_params=False)
eve_dir = os.path.join(results_dir, 'events')
rej_dir = os.path.join(results_dir, 'rejection-thresholds')
epo_dir = os.path.join(results_dir, 'epochs')
for _dir in (epo_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)
epoch_strategies = load_params(os.path.join('..', 'params', 'min_epochs.yaml'))
freq_bands = load_params(os.path.join('..', 'params', 'freq_bands.yaml'))

not_enough = defaultdict(set)

for subj in subjects:
    # load raw
    _dir = os.path.join(data_root, subj)
    raw_fname = f'{subj}_allclean_fil{lp_cut}_raw_sss.fif'
    raw_fpath = os.path.join(_dir, 'sss_pca_fif', raw_fname)
    eve_fpath = os.path.join(_dir, 'lists', f'ALL_{subj}-eve.lst')
    raw = mne.io.read_raw_fif(raw_fpath)
    # get the event dict from the auto-epoching done by mnefun
    epo_fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
    event_id = mne.read_epochs(os.path.join(_dir, 'epochs', epo_fname),
                               preload=False, proj=False, verbose=False
                               ).event_id
    # filter
    for freq_band, (l_freq, h_freq) in freq_bands.items():
        filtered_raw = (raw.copy()
                           .load_data()
                           .filter(l_freq, h_freq, n_jobs='cuda'))
        # loop over epoch lengths
        for epoch_dict in epoch_strategies:
            n_sec = int(epoch_dict["length"])
            slug = f'{subj}-{n_sec}sec'
            # load events
            eve_fpath = os.path.join(eve_dir, f'{slug}-eve.txt')
            events = mne.read_events(eve_fpath)
            # load precomputed autoreject params
            rej_fname = (f'{slug}-rej-thresh.yaml')
            rej_fpath = os.path.join(rej_dir, rej_fname)
            reject = load_params(rej_fpath)
            selection = reject.pop('selection')
            # epoch. Our raw sfreq is ridiculously high (over 3k), so we
            # decimate by a factor of 4 to save space/computation time.
            epochs = mne.Epochs(
                filtered_raw, events=events, event_id=event_id, decim=4,
                reject=None, baseline=None, preload=True,
                reject_by_annotation=False)
            # apply the results of autoreject
            epochs = epochs[selection]
            # truncate to same number of epochs per condition
            min_n_epochs = epoch_dict['n_min']
            new_events = np.empty((0, 3), dtype=int)
            event_codes = list(epochs.event_id.values())
            enough = True
            for code in event_codes:
                mask = epochs.events[:, -1] == code
                if sum(mask) < min_n_epochs:
                    not_enough[subj].add(n_sec)
                    enough = False
                new_events = np.vstack((
                    new_events, epochs.events[mask][:min_n_epochs]))
            if not enough:
                continue
            # restore temporal order of events
            new_events = new_events[np.argsort(new_events[:, 0])]
            equalized_selection = np.nonzero(
                np.in1d(epochs.events[:, 0], new_events[:, 0]))[0]
            epochs = epochs[equalized_selection]
            # save the filtered, decimated, equalized epochs
            epo_fname = (f'{subj}-{freq_band}-band-filtered-'
                         f'{n_sec}sec-epo.fif')
            epochs.save(os.path.join(epo_dir, epo_fname), overwrite=True)
        del filtered_raw

# note: `not_enough` values will be the same for all frequency bands, because
# they're based on the autoreject cutoffs that happen *before* filtering. Here
# they're sets, so duplicate entries from each frequency band don't matter.
not_enough = {key: sorted(val) for key, val in not_enough.items()}
with open(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'), 'w') as f:
    yaml.dump(not_enough, f)
