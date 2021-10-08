#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check which labels show correlated connectivity to see if we should merge some.

authors: Daniel McCloy
license: MIT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mne
import mne_connectivity
from f2f_helpers import load_paths, load_subjects, get_skip_regexp


def get_slug(subj, band, cond):
    return f'{subj}-{condition}-{freq_band}-band'


# flags
cov_type = 'erm'  # 'erm' or 'baseline'
freq_band = 'theta'
condition = 'allconds'
threshold_prop = 0.15
corr_thresh = 0.8

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
plot_dir = os.path.join(results_dir, 'figs', 'degree-correlations')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

subjects = load_subjects()

# colormap
cmap_kwargs = dict(h_neg=220, h_pos=10, sep=20)
cmap = sns.diverging_palette(s=30, l=67, as_cmap=True, **cmap_kwargs)
extremes = sns.diverging_palette(s=100, l=33, n=2, **cmap_kwargs)
cmap.set_extremes(under=extremes[0], over=extremes[1])

# load labels
regexp = get_skip_regexp()
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', regexp=regexp,
                                    subjects_dir=None)
label_names = [label.name for label in labels]

degree = np.zeros((0, 448), dtype=int)
for subj in subjects:
    # load connectivity
    slug = get_slug(subj, freq_band, condition)
    conn_fname = (f'{slug}-envelope-correlation.nc')
    conn_fpath = os.path.join(conn_dir, conn_fname)
    conn = mne_connectivity.read_connectivity(conn_fpath)
    this_degree = mne_connectivity.degree(conn, threshold_prop)
    degree = np.vstack((degree, this_degree))

corr = np.corrcoef(degree, rowvar=False)
corr_df = pd.DataFrame(corr, index=conn.names, columns=conn.names)
heatmap_kwargs = dict(cmap=cmap, vmin=-corr_thresh, vmax=corr_thresh)
figsize = (60, 60)
# make separate dataframes for each hemisphere (we'll never merge across hemis)
for hemi in ('lh', 'rh'):
    prefix = f'label-degree-correlations-{freq_band}-band'
    regex = f'.*-{hemi}'
    df = corr_df.filter(regex=regex, axis=0).filter(regex=regex, axis=1)
    # clustered heatmap
    cg = sns.clustermap(df, figsize=figsize, **heatmap_kwargs)
    fname = f'{prefix}-clustered-{hemi}.png'
    cg.fig.savefig(os.path.join(plot_dir, fname))
    # alphabetically-ordered heatmap
    fig = plt.figure(figsize=figsize)
    ax, cbar_ax = fig.subplots(
        1, 2, gridspec_kw=dict(width_ratios=(24, 1), wspace=0.1))
    ax = sns.heatmap(df, ax=ax, cbar_ax=cbar_ax, square=True, **heatmap_kwargs)
    fname = f'{prefix}-heatmap-{hemi}.png'
    fig.savefig(os.path.join(plot_dir, fname))

# print a list of correlated labels
lower_tri = np.tril(df.values, k=-1)
mask = np.abs(lower_tri) > corr_thresh
ixs = np.nonzero(mask)
pairs = np.array(list(zip(df.index.values[ixs[0]], df.columns.values[ixs[1]])))
