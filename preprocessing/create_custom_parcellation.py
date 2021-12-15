#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Make custom parcellation that incorporates our ROIs.

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
import mne
from matplotlib.colors import hex2color
from f2f_helpers import load_paths, load_params, get_skip_regexp


def make_rgba(hex_color, alpha=1.):
    return hex2color(hex_color) + (alpha,)


rng = np.random.default_rng(15485863)  # the one millionth prime

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
plot_dir = os.path.join(results_dir, 'figs', 'parcellations')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
skips_dict = load_params(os.path.join(param_dir, 'skip_labels.yaml'))
Brain = mne.viz.get_brain_class()

# setup colors
colors = dict(
    FEF='#882E72',
    IPS='#1965B0',
    TPJ='#7BAFDE',
    VFC='#4EB265',
    articulatory='#4EB265',  # roughly LH analog of VFC, so same color
    combinatorial='#CAE0AB',
    lexical='#F7F056',
    phonetics='#F4A736',
    phonology='#E8601C',
    premotor='#DC050C',
    sensorimotor='#72190E'
)

# load coarse parcellation (just need the names of major regions)
regexp = get_skip_regexp(skips_dict['aparc'])
aparc_labels = mne.read_labels_from_annot(
    surrogate, parc='aparc', regexp=regexp, subjects_dir=subjects_dir)
aparc_names = sorted(set(label.name.split('-')[0] for label in aparc_labels))
del aparc_labels

# load fine-grained parcellation
parcellation = 'aparc_sub'
regexp = get_skip_regexp(skips_dict[parcellation])
labels = mne.read_labels_from_annot(
    surrogate, parc=parcellation, regexp=regexp, subjects_dir=subjects_dir)
label_dict = {label.name: label for label in labels}

# create the ROI labels
roi_dict = load_params(os.path.join(param_dir, 'rois.yaml'))['hickok_corbetta']
roi_labels = list()
used_labels = dict()
for hemi, rois in roi_dict.items():
    used_labels[hemi] = list()
    for roi_name, constituent_labels in rois.items():
        this_roi = [label_dict[f'{label_name}-{hemi}']
                    for label_name in constituent_labels]
        this_roi = sum(this_roi[1:], start=this_roi[0])
        this_roi.name = f'{roi_name}-{hemi}'
        this_roi.color = make_rgba(colors[roi_name])
        roi_labels.append(this_roi)
        used_labels[hemi].extend(constituent_labels)
    # make sure we didn't use anything twice
    assert len(set(used_labels[hemi])) == len(used_labels[hemi])

# create "remainder" labels
remainder_labels = list()
for hemi in ('lh', 'rh'):
    for name in aparc_names:
        remainder = list()
        for key, label in label_dict.items():
            if key.split('-')[0] in used_labels[hemi]:
                continue
            elif key.startswith(name) and key.endswith(hemi):
                remainder.append(label)
        if len(remainder):
            remainder = sum(remainder[1:], start=remainder[0])
            remainder.name = f'{name}-{hemi}'
            # make other labels random colors
            rands = 1 - rng.random((3,))            # ensure always in (0, 1]
            rands = (rands + rands.mean()) / 2      # make more muted
            rands /= 10                             # make dark
            remainder.color = tuple(rands) + (1.,)  # add alpha channel
            remainder_labels.append(remainder)

# combine ROIs with remainders, and save
all_labels = roi_labels + remainder_labels
mne.write_labels_to_annot(
    all_labels, subject=surrogate, parc='hickok_corbetta',
    subjects_dir=subjects_dir, overwrite=True)
