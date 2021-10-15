#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sanity check for connectivity degree (beta band should show parietal activity).

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
import mne
from mne.minimum_norm import read_inverse_operator
import mne_connectivity
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params, get_skip_regexp


def get_slug(subj, band, cond):
    return f'{subj}-{condition}-{freq_band}-band'


# flags
cov_type = 'erm'  # 'erm' or 'baseline'
freq_band = 'beta'
condition = 'allconds'

# proportion of strongest edges to keep in the graph:
threshold_props = np.linspace(0.05, 0.5, 10)

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
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

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)

labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml'))

for parcellation, skips in labels_to_skip.items():
    regexp = get_skip_regexp(skips)
    for threshold_prop in threshold_props:
        grandavg_stc = None
        # load labels
        labels = mne.read_labels_from_annot(
            surrogate, parcellation, regexp=regexp, subjects_dir=subjects_dir)
        label_names = [label.name for label in labels]
        for subj in subjects:
            # load source space (from inverse)
            inv_fnames = dict(
                erm=f'{subj}-meg-erm{orientation_constraint}-inv.fif',
                baseline=(f'{subj}-{lp_cut}-sss-meg{orientation_constraint}'
                          '-inv.fif')
            )
            inv_fname = inv_fnames[cov_type]
            inv_fpath = os.path.join(data_root, subj, 'inverse', inv_fname)
            source_space = read_inverse_operator(inv_fpath)['src']
            # load connectivity
            slug = get_slug(subj, freq_band, condition)
            conn_fname = (f'{parcellation}-{slug}-envelope-correlation.nc')
            conn_fpath = os.path.join(conn_dir, conn_fname)
            conn = mne_connectivity.read_connectivity(conn_fpath)
            full_degree = mne_connectivity.degree(conn, threshold_prop)
            # convert to STC
            stc = mne.labels_to_stc(labels, full_degree)
            stc = stc.in_label(
                mne.Label(source_space[0]['vertno'], hemi='lh') +
                mne.Label(source_space[1]['vertno'], hemi='rh'))
            # aggregate
            if grandavg_stc is None:
                grandavg_stc = stc
            else:
                grandavg_stc.data += stc.data
            del conn, full_degree
        grandavg_stc.data /= len(subjects)

        # plot grand average STC
        brain = grandavg_stc.plot(
            clim=dict(kind='percent', lims=[75, 85, 95]),
            colormap='plasma',
            subjects_dir=subjects_dir,
            views=['lateral', 'medial', 'dorsal'], hemi='split',
            smoothing_steps='nearest', time_label=f'{freq_band} band',
            size=(900, 1200), background='w', title=slug)
        # save
        slug = get_slug('grandavg', freq_band, condition)
        img_fname = (f'{parcellation}-{slug}-degree-thresh_'
                     f'{threshold_prop:.02f}.png')
        brain.save_image(os.path.join(plot_dir, img_fname))
        brain.close()
