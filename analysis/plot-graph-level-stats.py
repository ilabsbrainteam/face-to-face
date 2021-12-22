#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot graph-level stats.

authors: Daniel McCloy
license: MIT
"""

from functools import reduce
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from f2f_helpers import load_paths, load_params

# flags
freq_band = 'theta'
n_sec = 7  # epoch duration to use
sns.set(font_scale=0.8)
cluster_plot = False

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
xarray_dir = os.path.join(results_dir, 'xarrays')
plot_dir = os.path.join(results_dir, 'figs', 'graph-metrics')

# load ROI info
roi_edge_dict = load_params(os.path.join(param_dir, 'roi-edges.yaml'))

for parcellation, roi_edges in roi_edge_dict.items():
    roi_nodes = sorted(set(reduce(tuple.__add__, roi_edges)))
    for use_edge_rois in (False, True):
        roi = 'roi' if use_edge_rois else 'all'
        slug = f'{parcellation}-{n_sec}sec-{freq_band}-band-{roi}-edges'
        # load lambda squared
        fname = f'{slug}-lambda-sq.nc'
        lambda_sq = xr.load_dataarray(os.path.join(xarray_dir, fname))
        # sort rows/cols by hemisphere
        df = np.abs(lambda_sq).to_pandas()
        sorted_regions = (df.index.to_series()
                            .str.split('-', expand=True)
                            .sort_values([1, 0])
                            .index.tolist())
        df = df.loc[sorted_regions, sorted_regions]
        if use_edge_rois:
            ixs = np.nonzero(np.in1d(sorted_regions, roi_nodes)) * 2
            df = df.iloc[ixs]
        # plot relative contribution of each edge to the difference
        clust = '-clustered' if cluster_plot else ''
        fname = (f'{slug}-attend_minus_ignore{clust}.pdf')
        figsize = (8, 8) if use_edge_rois else (32, 32)
        if cluster_plot:
            cg = sns.clustermap(df, figsize=figsize)
            fig = cg.fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(df, square=True, ax=ax)
        fig.savefig(os.path.join(plot_dir, fname))
        del fig
