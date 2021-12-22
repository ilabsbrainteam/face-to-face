#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot node-level stats.

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from f2f_helpers import load_paths, load_params, yamload

# flags
freq_band = 'theta'
n_sec = 7  # epoch duration to use
scopes = ('roi', 'all')

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
xarray_dir = os.path.join(results_dir, 'xarrays')
stats_dir = os.path.join(results_dir, 'stats')
plot_dir = os.path.join(results_dir, 'figs', 'node-metrics')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
all_roi_dicts = load_params(os.path.join(param_dir, 'rois.yaml'))

# for plotting
medial_wall_labels = {parc: (
        'cuneus', 'precuneus', 'paracentral', 'superiorfrontal',
        'medialorbitofrontal', 'rostralanteriorcingulate',
        'caudalanteriorcingulate', 'posteriorcingulate', 'isthmuscingulate',
        'parahippocampal', 'entorhinal', 'fusiform', 'lingual',
        'pericalcarine')
    for parc in all_roi_dicts}

# containers
Brain = mne.viz.get_brain_class()

for parcellation in all_roi_dicts:
    for scope in scopes:
        slug = f'{parcellation}-{n_sec}sec-{freq_band}-band-{scope}-edges'
        fname = f'{slug}-node-level-stats.yaml'
        with open(os.path.join(stats_dir, fname)) as f:
            stats = yamload(f)
        for thresh_kind in stats:
            for metric in stats[thresh_kind]:
                regions = list(stats[thresh_kind][metric])
                # plot signifs
                brain = Brain(
                    surrogate, hemi='split', surf='inflated', size=(1200, 900),
                    cortex='low_contrast', views=['lateral', 'medial'],
                    background='white', subjects_dir=subjects_dir)
                regexp = '|'.join(regions)
                # avoid empty regexp loading all labels
                signif_labels = (
                    list() if not len(regions) else
                    mne.read_labels_from_annot(
                        surrogate, parcellation, regexp=regexp,
                        subjects_dir=subjects_dir)
                    )
                # prevent text overlap
                text_bookkeeping = {(row, col): list() for row in (0, 1)
                                    for col in (0, 1)}
                # draw labels and add label names
                for label in signif_labels:
                    brain.add_label(label, alpha=0.5)
                    brain.add_label(label, borders=True)
                    col = int(label.hemi == 'rh')
                    row = int(label.name.rsplit('-')[0].rsplit('_')[0]
                              in medial_wall_labels[parcellation])
                    y = 0.02 + 0.06 * len(text_bookkeeping[(row, col)])
                    text_bookkeeping[(row, col)].append(label.name)
                    brain.add_text(
                        0.05, y, text=label.name.rsplit('-')[0],
                        name=label.name, row=row, col=col, color=label.color,
                        font_size=12)
                fname = (f'{parcellation}-{n_sec}sec-{metric}-'
                         f'{thresh_kind}-signif-{scope}-labels.png')
                brain.save_image(os.path.join(plot_dir, fname))
                brain.close()
                del brain
