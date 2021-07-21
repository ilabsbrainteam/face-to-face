#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mnefun preprocessing of face-to-face data (for connectivity analysis).

authors: Daniel McCloy
license: MIT
"""

import mnefun
from f2f_helpers import load_paths, load_subjects
from f2f_score import f2f_score

# load general params
data_root, subjects_dir, _ = load_paths()
subjects = load_subjects()

# load mnefun params from YAML
params = mnefun.read_params('mnefun_params.yaml')

# set additional params: general
params.score = f2f_score
params.subjects_dir = subjects_dir
params.subjects = subjects
params.subject_indices = list(range(len(params.subjects)))

# set additional params: report
kwargs = dict(analysis='Conditions', cov=f'%s-{params.lp_cut}-sss-cov.fif')
params.report['whitening'] = [dict(name=c, **kwargs) for c in params.in_names]

# run it
mnefun.do_processing(
    params,
    fetch_raw=False,      # go get the Raw files
    do_sss=True,          # tSSS / maxwell filtering
    do_score=True,        # run scoring function to extract events
    gen_ssp=True,         # create SSP projectors
    apply_ssp=True,       # apply SSP projectors
    write_epochs=True,    # epoching & filtering
    gen_covs=True,        # make covariance
    gen_fwd=True,         # generate fwd model
    gen_inv=True,         # generate inverse
    gen_report=True,      # print report
    print_status=True     # show status
)
