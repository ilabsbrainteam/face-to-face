#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copy custom parcellation from our surrogate brain to each subj.

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from f2f_helpers import load_paths, load_params, load_subjects

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))

# load the annotation from surrogate
labels = mne.read_labels_from_annot(
    surrogate, 'f2f_custom', subjects_dir=subjects_dir)

# copy to each subj
for subj in subjects:
    # morph labels
    new_labels = mne.morph_labels(
        labels, subject_to=subj, subject_from=surrogate,
        subjects_dir=subjects_dir)
    # rename labels
    for label in new_labels:
        label.name = f'f2f-{label.name}'
    mne.write_labels_to_annot(
        labels=new_labels, subject=subj, parc='f2f_custom',
        subjects_dir=subjects_dir, overwrite=True)
    # save labels
    for label in new_labels:
        fname = f'{label.hemi}.{label.name.rsplit("-", maxsplit=1)[0]}.label'
        fpath = os.path.join(subjects_dir, subj, 'label', fname)
        mne.write_label(fpath, label)
