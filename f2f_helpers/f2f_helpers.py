#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
from functools import partial
import mne

paramdir = os.path.join('..', 'params')
yamload = partial(yaml.load, Loader=yaml.FullLoader)


def load_paths(include_inv_params=True):
    """Load necessary filesystem paths."""
    with open(os.path.join(paramdir, 'paths.yaml'), 'r') as f:
        paths = yamload(f)
    if include_inv_params:
        params = load_params(os.path.join(paramdir, 'inverse_params.yaml'))
        _dir = f"{params['orientation_constraint']}-{params['estimate_type']}"
        paths['results_dir'] = os.path.join(paths['results_dir'], _dir)
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


def get_skip_regexp(regions=(), skip_unknown=True):
    """Convert an iterable of region names to a label regexp excluding them."""
    unknown = ('unknown', r'\?\?\?')
    if skip_unknown:
        regions = regions + unknown
    skip_labels = tuple(f'{region}-{hemi}'
                        for region in regions for hemi in ('lh', 'rh'))
    if len(skip_labels):
        return f"(?!{'|'.join(skip_labels)})"


def get_roi_labels(subject, param_dir, parc='aparc', merge=None):
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
        'parsorbitalis': '#90C987',  # green
        'parsopercularis': '#90C987',  # green
        'parstriangularis': '#90C987',  # green
    }
    # load ROIs
    rois = load_params(os.path.join(param_dir, 'rois.yaml'))
    rois_to_merge = list()
    if merge is not None:
        for labels_to_merge in merge.values():
            rois_to_merge.extend(labels_to_merge)
    rois = set(rois) - set(rois_to_merge)
    roi_regexp = '|'.join(rois)
    # get regular labels
    labels = mne.read_labels_from_annot(
        subject, parc=parc, subjects_dir=subjects_dir,
        regexp=roi_regexp)
    # set label colors
    for label in labels:
        label.color = label_colors[label.name.rsplit('-', maxsplit=1)[0]]
    # get merged labels
    if merge is not None:
        for h in ('lh', 'rh'):
            for name, labels_to_merge in merge.items():
                _regexp = '|'.join(f'{label}-{h}' for label in labels_to_merge)
                _labels = mne.read_labels_from_annot(
                    subject, parc=parc, subjects_dir=subjects_dir,
                    regexp=_regexp)
                merged_label = sum(_labels[1:], _labels[0])
                merged_label.name = f'{name}-{h}'
                # set merged label color
                color_key = _labels[0].name.rsplit('-', maxsplit=1)[0]
                merged_label.color = label_colors[color_key]
                labels.append(merged_label)
    return labels
