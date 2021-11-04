#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check which labels show correlated connectivity to see if we should merge some.

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
from matplotlib.pyplot import close as closefig
import pandas as pd
import seaborn as sns
import mne
import mne_connectivity
from f2f_helpers import (load_paths, load_subjects, load_params, get_slug,
                         get_skip_regexp)

# flags
freq_band = 'theta'
condition = 'allconds'
parcellation = 'aparc'
threshold_prop = 0.15
corr_thresh = 0.8

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
epo_dir = os.path.join(results_root_dir, 'epochs')
param_dir = os.path.join('..', 'params')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
plot_dir = os.path.join(results_dir, 'figs', 'degree-correlations')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
epoch_strategies = load_params(os.path.join(param_dir, 'min_epochs.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))

inv_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
cov_type = inv_params['cov_type']  # 'erm' or 'baseline'

# colormap
cmap_kwargs = dict(h_neg=220, h_pos=10, sep=20)
cmap = sns.diverging_palette(s=30, l=67, as_cmap=True, **cmap_kwargs)
extremes = sns.diverging_palette(s=100, l=33, n=2, **cmap_kwargs)
cmap.set_extremes(under=extremes[0], over=extremes[1])

# load labels
labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml'))

for parcellation, skips in labels_to_skip.items():
    regexp = get_skip_regexp(skips)
    labels = mne.read_labels_from_annot(
        surrogate, parc=parcellation, regexp=regexp, subjects_dir=subjects_dir)

    # loop over epoch lengths
    for epoch_dict in epoch_strategies:
        degree = np.zeros((0, len(labels)), dtype=int)
        n_sec = int(epoch_dict["length"])
        for subj in subjects:
            # check if we should skip
            if subj in excludes and n_sec in excludes[subj]:
                continue
            # load connectivity
            slug = get_slug(subj, freq_band, condition, parcellation)
            conn_fname = (f'{slug}-{n_sec}sec-envelope-correlation.nc')
            conn_fpath = os.path.join(conn_dir, conn_fname)
            conn = mne_connectivity.read_connectivity(conn_fpath)
            this_degree = mne_connectivity.degree(conn, threshold_prop)
            degree = np.vstack((degree, this_degree))

        corr = np.corrcoef(degree, rowvar=False)
        corr_df = pd.DataFrame(corr, index=conn.names, columns=conn.names)
        heatmap_kwargs = dict(cmap=cmap, vmin=-corr_thresh, vmax=corr_thresh)
        inches = np.round(len(labels), -1) // 5
        figsize = (inches, inches)
        # make separate dfs for each hemisphere (we never merge across hemis)
        for hemi in ('lh', 'rh'):
            prefix = (f'{parcellation}-label-degree-correlations-'
                      f'{n_sec}sec-{freq_band}-band')
            regex = f'.*-{hemi}'
            df = (corr_df.filter(regex=regex, axis=0)
                         .filter(regex=regex, axis=1))
            # clustered heatmap
            cg = sns.clustermap(df, figsize=figsize, **heatmap_kwargs)
            fname = f'{prefix}-clustered-{hemi}.png'
            cg.fig.savefig(os.path.join(plot_dir, fname))
            closefig(cg.fig)
