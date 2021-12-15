#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sanity check for parcellation morphing.

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from f2f_helpers import load_paths, load_params, get_skip_regexp

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
plot_dir = os.path.join(results_dir, 'figs', 'parcellations')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
parcellations = ('hickok_corbetta', 'aparc', 'aparc_sub', 'HCPMMP1_combined',
                 'HCPMMP1')
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
Brain = mne.viz.get_brain_class()

# plot each parcellation
for parcellation in parcellations:
    for surf in ('pial', 'inflated'):
        brain = Brain(
            surrogate, hemi='split', surf=surf, size=(1200, 900),
            cortex='low_contrast', views=['lateral', 'medial'],
            background='white', subjects_dir=subjects_dir)
        # load all labels except "unknown" or "???"
        regexp = get_skip_regexp()
        labels = mne.read_labels_from_annot(
            surrogate, parcellation, regexp=regexp, subjects_dir=subjects_dir)
        for label in labels:
            brain.add_label(label, alpha=0.4)
            brain.add_label(label, borders=True)
        brain.save_image(os.path.join(plot_dir, f'{parcellation}-{surf}.png'))
        brain.close()
        del brain
