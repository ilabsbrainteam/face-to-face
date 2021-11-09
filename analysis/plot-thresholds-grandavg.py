#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sanity check for connectivity degree (ideally beta band should show parietal
activity, though this is technically not "resting state").

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
import mne
from mne.minimum_norm import read_inverse_operator
import mne_connectivity
import mnefun
from f2f_helpers import (load_paths, load_subjects, load_params, get_slug,
                         get_skip_regexp)

# flags
freq_band = 'beta'
condition = 'allconds'

# proportion of strongest edges to keep in the graph:
threshold_props = np.linspace(0.05, 0.5, 10)

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
param_dir = os.path.join('..', 'params')
epo_dir = os.path.join(results_root_dir, 'epochs')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
plot_dir = os.path.join(results_dir, 'figs', 'thresh-prop')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
inv_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
orientation_constraint = (
    '' if inv_params['orientation_constraint'] == 'loose' else
    f"-{inv_params['orientation_constraint']}")
cov_type = inv_params['cov_type']  # 'erm' or 'baseline'

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)
labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml'))
epoch_strategies = load_params(os.path.join(param_dir, 'min_epochs.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))

# precompute how many excluded subjs per epoch length
for epoch_dict in epoch_strategies:
    epoch_dict['n_excludes'] = 0
    n_sec = int(epoch_dict["length"])
    for exclude in excludes.values():
        if n_sec in exclude:
            epoch_dict['n_excludes'] += 1

for parcellation, skips in labels_to_skip.items():
    regexp = get_skip_regexp(skips)
    if parcellation == 'f2f_custom':
        regexp = f'f2f-{regexp}'
    for threshold_prop in threshold_props:
        grandavg_stc = dict()
        for subj in subjects:
            # check if we should skip
            if (subj in excludes and
                    len(excludes[subj]) == len(epoch_strategies)):
                continue
            # load labels
            labels = mne.read_labels_from_annot(
                subj, parcellation, regexp=regexp, subjects_dir=subjects_dir)
            # load source space (from inverse)
            inv_fnames = dict(
                erm=f'{subj}-meg-erm{orientation_constraint}-inv.fif',
                baseline=(f'{subj}-{lp_cut}-sss-meg{orientation_constraint}'
                          '-inv.fif')
            )
            inv_fname = inv_fnames[cov_type]
            inv_fpath = os.path.join(data_root, subj, 'inverse', inv_fname)
            source_space = read_inverse_operator(inv_fpath)['src']
            # loop over epoch lengths
            for epoch_dict in epoch_strategies:
                # check if we should skip
                n_sec = int(epoch_dict["length"])
                if subj in excludes and n_sec in excludes[subj]:
                    continue
                # load connectivity
                slug = get_slug(subj, freq_band, condition, parcellation)
                conn_fname = (f'{slug}-{n_sec}sec-envelope-correlation.nc')
                conn_fpath = os.path.join(conn_dir, conn_fname)
                conn = mne_connectivity.read_connectivity(conn_fpath)
                full_degree = mne_connectivity.degree(conn, threshold_prop)
                # convert to STC
                stc = mne.labels_to_stc(labels, full_degree, subject=subj,
                                        src=source_space)
                # aggregate
                if grandavg_stc.get(n_sec, None) is None:
                    grandavg_stc[n_sec] = stc
                else:
                    grandavg_stc[n_sec].data += stc.data
                del conn, full_degree

        # finish aggregation over subjects
        for epoch_dict in epoch_strategies:
            n_sec = int(epoch_dict["length"])
            n_excludes = epoch_dict['n_excludes']
            grandavg_stc[n_sec].data /= (len(subjects) - n_excludes)

            # plot grand average STC for each epoch duration
            brain = grandavg_stc[n_sec].plot(
                clim=dict(kind='percent', lims=[75, 85, 95]),
                colormap='plasma',
                subjects_dir=subjects_dir,
                views=['lateral', 'medial', 'dorsal'], hemi='split',
                smoothing_steps='nearest',
                time_label=f'{freq_band} band, {n_sec}-second epochs',
                size=(900, 1200), background='w', title=slug)
            # save
            slug = get_slug('grandavg', freq_band, condition, parcellation)
            img_fname = (f'{slug}-{n_sec}sec-degree-thresh_'
                         f'{threshold_prop:.02f}.png')
            brain.save_image(os.path.join(plot_dir, img_fname))
            brain.close()
