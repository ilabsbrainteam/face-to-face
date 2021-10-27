#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do peak-to-peak epoch rejections for the different epoch lengths/overlaps
in the face-to-face data.

authors: Daniel McCloy
license: MIT
"""

import os
import yaml
from collections import Counter
import numpy as np
import pandas as pd
import mne
from autoreject import get_rejection_threshold
import mnefun
from f2f_helpers import (load_paths, load_subjects, load_params,
                         compute_epoch_offsets)

# config paths
data_root, subjects_dir, results_dir = load_paths(include_inv_params=False)
eve_dir = os.path.join(results_dir, 'events')
rej_dir = os.path.join(results_dir, 'rejection-thresholds')
for _dir in (eve_dir, rej_dir):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)
orig_dur = mnefun_params.tmax - mnefun_params.tmin
epoch_strategies = load_params(os.path.join('..', 'params', 'min_epochs.yaml'))
freq_bands = load_params(os.path.join('..', 'params', 'freq_bands.yaml'))

# container
df = pd.DataFrame()

for subj in subjects:
    # load raw and events
    _dir = os.path.join(data_root, subj)
    raw_fname = f'{subj}_allclean_fil{lp_cut}_raw_sss.fif'
    raw_fpath = os.path.join(_dir, 'sss_pca_fif', raw_fname)
    eve_fpath = os.path.join(_dir, 'lists', f'ALL_{subj}-eve.lst')
    raw = mne.io.read_raw_fif(raw_fpath, preload=False)
    events = mne.read_events(eve_fpath)
    # get the event dict from the auto-epoching done by mnefun
    epo_fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
    event_id = mne.read_epochs(os.path.join(_dir, 'epochs', epo_fname),
                               preload=False, proj=False, verbose=False
                               ).event_id
    # epoch without filtering, so that noisy trials (that we ought to reject)
    # don't spuriously seem "clean" because of the filtering
    for epoch_dict in epoch_strategies:
        slug = f'{subj}-{int(epoch_dict["length"])}sec'
        offset_samps = compute_epoch_offsets(
            orig_dur=orig_dur, sfreq=raw.info['sfreq'], **epoch_dict)
        # regenerate events array with potentially shorter epochs
        new_events = np.empty((0, 3), dtype=int)
        for row in events:
            offset_events = np.hstack((
                offset_samps + row[0],
                np.zeros_like(offset_samps),
                np.full_like(offset_samps, row[-1])
            ))
            new_events = np.vstack((new_events, offset_events))
        # save the re-spaced events array
        new_eve_fpath = os.path.join(eve_dir, f'{slug}-eve.txt')
        mne.write_events(new_eve_fpath, new_events)
        # now epoch w/ autoreject
        epochs = mne.Epochs(
            raw, events=new_events, event_id=event_id, reject=None,
            baseline=None, preload=True)
        reject = get_rejection_threshold(epochs)
        epochs.drop_bad(reject)
        # cast reject values to python float for cleaner YAML writes, and save.
        # Also save epochs.selection, which we will use later for dropping
        # epochs after filtering
        reject = {key: float(val) for key, val in reject.items()}
        reject['selection'] = [int(n) for n in epochs.selection]
        rej_fname = (f'{slug}-rej-thresh.yaml')
        rej_fpath = os.path.join(rej_dir, rej_fname)
        with open(rej_fpath, 'w') as f:
            yaml.dump(reject, f)
        # aggregate the results
        n_good = dict(Counter(epochs.events[:, -1]))
        this_row = pd.DataFrame(dict(subj=[subj], dur=[epoch_dict['length']]))
        for cond, _id in epochs.event_id.items():
            this_row[cond] = [n_good[_id]]
        df = pd.concat((df, this_row))

# sort by smallest value in either of our condition columns
columns = ['attend', 'ignore']
df = (df.assign(min=df[columns].values.min(axis=1),
                max=df[columns].values.max(axis=1))
        .sort_values(['dur', 'min', 'max'])
        .drop(columns=['min', 'max']))
# save
fpath = os.path.join(results_dir, 'epochs-retained-after-autoreject.csv')
df.to_csv(fpath, index=False)
