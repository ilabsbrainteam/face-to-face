#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rescale MRI labels without re-running coregistration (run-once fix for bad
labels).

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from f2f_helpers import load_paths, load_subjects, load_params

data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
subjects = load_subjects()

for subj in subjects:
    mne.scale_labels(
        subj, overwrite=True, subjects_dir=subjects_dir)
