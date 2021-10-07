#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Envelope correlation of face-to-face data, in several bands.

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from f2f_helpers import load_paths, load_subjects, get_skip_regexp

# "aparc" already exists for all subjects
parcellations = ('aparc_sub', 'HCPMMP1_combined', 'HCPMMP1')

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')

# load other config values
subjects = load_subjects()

# loop over parcellations
for parcellation in parcellations:
    # load labels
    regexp = get_skip_regexp()
    labels = mne.read_labels_from_annot(
        'fsaverage', parcellation, regexp=regexp, subjects_dir=subjects_dir)
    # morph labels
    for subj in subjects:
        this_labels = mne.morph_labels(
            labels, subject_to=subj, subject_from='fsaverage',
            subjects_dir=subjects_dir)
        mne.write_labels_to_annot(
            labels=this_labels, subject=subj, parc=parcellation,
            subjects_dir=subjects_dir, overwrite=False)
