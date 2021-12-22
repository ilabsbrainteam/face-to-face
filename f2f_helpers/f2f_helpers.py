#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
from functools import partial
import numpy as np

paramdir = os.path.join('..', 'params')
yamload = partial(yaml.load, Loader=yaml.FullLoader)


def load_params(fname):
    """Load parameters from YAML file."""
    with open(fname, 'r') as f:
        params = yamload(f)
    return params


def load_paths(include_inv_params=True):
    """Load necessary filesystem paths."""
    paths = load_params(os.path.join(paramdir, 'paths.yaml'))
    if include_inv_params:
        params = load_params(os.path.join(paramdir, 'inverse_params.yaml'))
        _dir = f"{params['orientation_constraint']}-{params['estimate_type']}"
        paths['results_dir'] = os.path.join(paths['results_dir'], _dir)
    return paths['data_root'], paths['subjects_dir'], paths['results_dir']


def load_subjects(skip=True):
    """Load subject IDs."""
    subjects = load_params(os.path.join(paramdir, 'subjects.yaml'))
    # skip bad subjects
    if skip:
        skips = load_params(os.path.join(paramdir, 'skip_subjects.yaml'))
        subjects = sorted(set(subjects) - set(skips))
    return subjects


def get_skip_regexp(regions=(), skip_unknown=True, prefix=''):
    """Convert an iterable of region names to a label regexp excluding them."""
    unknown = ('unknown', r'\?\?\?')
    if skip_unknown:
        regions = regions + unknown
    if prefix:
        regions = tuple(f'{prefix}-{region}' for region in regions)
    if len(regions):
        return f"(?!{'|'.join(regions)})"


def get_slug(subject, freq_band, condition, parcellation=None):
    """Assemble a filename slug from experiment parameters."""
    parcellation = '' if parcellation is None else f'{parcellation}-'
    return f'{parcellation}{subject}-{condition}-{freq_band}-band'


def compute_epoch_offsets(length, overlap, orig_dur, sfreq, n_min=None):
    """Compute sample offsets for chopping a long epoch into shorter ones.

    Note: n_min is ignored, included in signature only because it's part of the
    dicts that get passed in by dict unpacking.
    """
    spacing = length - overlap
    if spacing == 0:
        n_epochs = 1
    else:
        n_epochs = int((orig_dur - length) // spacing) + 1
    offsets = np.linspace(0, orig_dur - length, n_epochs)
    offset_samps = np.rint(offsets * sfreq).astype(int)
    # make sure subsequent trials don't overlap too much
    assert np.all(np.diff(offset_samps) >= np.floor(spacing * sfreq))
    # make 2D and vertical
    return np.stack([offset_samps], axis=1)
