#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
from functools import partial
import mne

paramdir = os.path.join('..', 'params')
yamload = partial(yaml.load, Loader=yaml.FullLoader)


def load_paths():
    """Load necessary filesystem paths."""
    with open(os.path.join(paramdir, 'paths.yaml'), 'r') as f:
        paths = yamload(f)
    return paths['data_root'], paths['subjects_dir'], paths['results_dir']


def load_subjects(skip=True):
    """Load subject IDs."""
    with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
        subjects = yamload(f)
    # skip bad subjects
    if skip:
        with open(os.path.join(paramdir, 'skip_subjects.yaml'), 'r') as f:
            skips = yamload(f)
        subjects = sorted(set(subjects) - set(skips))
    return subjects


def load_params(fname):
    """Load parameters from YAML file."""
    with open(fname, 'r') as f:
        params = yamload(f)
    return params


def get_roi_labels(subject, param_dir):
    """Load ROI labels for the given subject."""
    _, subjects_dir, _ = load_paths()
    label_colors = {
        'caudalanteriorcingulate': '#D1BBD7',  # purple, light
        'inferiorparietal': '#4EB265',  # green, dark
        'inferiortemporal': '#AE76A3',  # purple
        'insula': '#CAE0AB',  # green, light
        'lateralorbitofrontal': '#882E72',  # purple, dark
        'medialorbitofrontal': '#F7F056',  # yellow
        'middletemporal': '#1965B0',  # blue, dark
        'posteriorcingulate': '#F4A736',  # orange, light
        'rostralanteriorcingulate': '#5289C7',  # blue
        'superiorparietal': '#E8601C',  # orange
        'superiortemporal': '#7BAFDE',  # blue, light
        'temporalpole': '#DC050C',  # red
        'inferiorfrontal': '#90C987',  # green pars{orbital|operc|triangular}is
    }
    # load ROIs
    rois = load_params(os.path.join(param_dir, 'rois.yaml'))
    rois_to_merge = ('parsorbitalis', 'parsopercularis', 'parstriangularis')
    rois = set(rois) - set(rois_to_merge)
    roi_regexp = '|'.join(rois)
    # get regular labels
    labels = mne.read_labels_from_annot(
        subject, parc='aparc', subjects_dir=subjects_dir,
        regexp=roi_regexp)
    # get merged labels
    for h in ('lh', 'rh'):
        _regexp = '|'.join((f'{roi}-{h}' for roi in rois_to_merge))
        _labels = mne.read_labels_from_annot(
            subject, parc='aparc', subjects_dir=subjects_dir,
            regexp=_regexp)
        merged_label = sum(_labels[1:], _labels[0])
        merged_label.name = f'inferiorfrontal-{h}'
        labels.append(merged_label)
    # set label colors
    for label in labels:
        label.color = label_colors[label.name.rsplit('-', maxsplit=1)[0]]
    return labels
