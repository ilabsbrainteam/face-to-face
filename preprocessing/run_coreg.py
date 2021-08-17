#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mnefun preprocessing of face-to-face data (for connectivity analysis).

authors: Daniel McCloy
license: MIT
"""

import os
from mne.gui import coregistration
from f2f_helpers import load_paths, load_subjects, load_params

# load general params
data_root, subjects_dir, _ = load_paths()
param_dir = os.path.join('..', 'params')
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))

# generate the MRI config files for scaling surrogate MRI to individual
# subject's digitization points.
#
#     - Run once before preprocessing
#     - Fit fiducials first, then ICP; use uniform scaling
#     - Deselect all subject-saving options (scale label files / prepare BEM)
#     - Save

for subject in subjects:
    raw_fname = os.path.join(data_root, subject, 'raw_fif',
                             f'{subject}_raw.fif')
    coregistration(inst=raw_fname,
                   subject=surrogate,
                   subjects_dir=subjects_dir,
                   guess_mri_subject=False,
                   mark_inside=True)
