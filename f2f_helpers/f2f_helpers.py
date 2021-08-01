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


def scale_mri(subject, subjects_dir, subject_from, target_file):
    """Scale surrogate MRI to approximate subject headshape."""
    # skip if already exists (subj-specific MRI or past surrogate scaling)
    target_path = os.path.join(subjects_dir, subject, 'mri', target_file)
    if os.path.exists(target_path):
        return
    # read scaling config
    config = mne.coreg.read_mri_cfg(subject, subjects_dir)
    assert config.pop('n_params') in (1, 3)
    assert config['subject_from'] == subject_from
    # do MRI scaling
    mne.coreg.scale_mri(subject_from=subject_from,
                        subject_to=subject,
                        subjects_dir=subjects_dir,
                        scale=config['scale'],
                        labels=False,
                        annot=False,
                        overwrite=True)
    # make BEM solution
    bem_in = os.path.join(
        subjects_dir, subject, 'bem', f'{subject}-5120-bem.fif')
    bem_out = os.path.join(
        subjects_dir, subject, 'bem', f'{subject}-5120-bem-sol.fif')
    solution = mne.make_bem_solution(bem_in)
    mne.write_bem_solution(bem_out, solution)
