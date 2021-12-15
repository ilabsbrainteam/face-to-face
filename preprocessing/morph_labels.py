#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Morph labels from fsaverage to our surrogate brain.

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from f2f_helpers import load_paths, load_params

# "hickok_corbetta" already exists for surrogate
parcellations = ('aparc', 'aparc_sub', 'HCPMMP1_combined', 'HCPMMP1')

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')

# load other config values
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))

# loop over parcellations
for parcellation in parcellations:
    # load all labels (no skips)
    labels = mne.read_labels_from_annot(
        'fsaverage', parcellation, subjects_dir=subjects_dir)
    # morph labels
    new_labels = mne.morph_labels(
        labels, subject_to=surrogate, subject_from='fsaverage',
        subjects_dir=subjects_dir)
    mne.write_labels_to_annot(
        labels=new_labels, subject=surrogate, parc=parcellation,
        subjects_dir=subjects_dir, overwrite=True)
    # save labels
    for label in new_labels:
        fname = f'{label.hemi}.{label.name.rsplit("-", maxsplit=1)[0]}.label'
        fpath = os.path.join(subjects_dir, surrogate, 'label', fname)
        mne.write_label(fpath, label)
