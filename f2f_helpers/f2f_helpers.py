#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
from functools import partial

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
