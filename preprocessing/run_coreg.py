#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mnefun preprocessing of face-to-face data (for connectivity analysis).

authors: Daniel McCloy
license: MIT
"""

import os
import mne
from f2f_helpers import load_paths, load_subjects, load_params

# load general params
data_root, subjects_dir, _ = load_paths()
param_dir = os.path.join('..', 'params')
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))

scaling_already_done = True

# generate the MRI config files for scaling surrogate MRI to individual
# subject's digitization points. Then scale the MRI and make the BEM solution.
for subject in subjects:
    if scaling_already_done:
        continue
    print('## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ##')
    print(f'##               NOW SCALING SUBJECT {subject}                ##')
    print('## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ## # ##')
    raw_fname = os.path.join(data_root, subject, 'raw_fif',
                             f'{subject}_raw.fif')
    info = mne.io.read_info(raw_fname)
    coreg = mne.coreg.Coregistration(
        info, subject=surrogate, subjects_dir=subjects_dir)
    coreg.set_scale_mode('3-axis')
    coreg.set_fid_match('matched')
    coreg.fit_icp(n_iterations=30)
    # this step takes a while
    mne.scale_mri(subject_from=surrogate, subject_to=subject,
                  scale=coreg.scale, overwrite=True, subjects_dir=subjects_dir,
                  verbose=True)
    # make BEM solution
    bem_in = os.path.join(
        subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem.fif')
    bem_out = os.path.join(
        subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')
    solution = mne.make_bem_solution(bem_in)
    mne.write_bem_solution(bem_out, solution)

# QC the coregistrations
for subject in subjects:
    this_subj_dir = os.path.join(data_root, subject, 'raw_fif')
    raw_fname = os.path.join(this_subj_dir, f'{subject}_raw.fif')
    trans_fname = os.path.join(this_subj_dir, f'{subject}-trans.fif')
    info = mne.io.read_info(raw_fname)
    trans = mne.read_trans(trans_fname)
    print(subject)
    mne.viz.plot_alignment(
        info=info, trans=trans, subject=subject, subjects_dir=subjects_dir,
        surfaces=dict(head=0.9), dig=True, mri_fiducials=True, meg=False)
    # run coregistration GUI to do it manually and compare
    mne.gui.coregistration(inst=raw_fname,
                           subject=surrogate,
                           subjects_dir=subjects_dir,
                           guess_mri_subject=False,
                           mark_inside=True)
    response = input(f'You just saw {subject}; press <ENTER> to continue or X '
                     'to quit\n')
    if response.lower().startswith('x'):
        break
