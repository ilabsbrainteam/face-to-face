#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correlate behavioral data with graph metrics.

authors: Daniel McCloy
license: MIT
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from f2f_helpers import load_paths, load_params

# flags
freq_band = 'theta'
parcellation = 'f2f_custom'
n_sec = 7  # epoch duration to use

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
xarray_dir = os.path.join(results_dir, 'xarrays')
beh_dir = os.path.join('..', 'behavioral')
plot_dir = os.path.join(results_dir, 'figs', 'behavior-correlations')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load ROIs
roi_dict = load_params(os.path.join(param_dir, 'rois.yaml'))[parcellation]
roi_names = list()
for hemi, roi_name_dict in roi_dict.items():
    for roi_name in roi_name_dict:
        roi_names.append(f'{roi_name}-{hemi}')

# xarray filenames
slug = f'{parcellation}-{n_sec}sec-{freq_band}-band'
graph_fname = f'{slug}-graph-metrics.nc'
node_fname = f'{slug}-all-edges-node-metrics.nc'

# load behavioral data
beh = pd.read_csv(os.path.join(beh_dir, 'f2f-behavioral-data.csv'))
# load graph metrics
graph_metrics = xr.load_dataarray(os.path.join(xarray_dir, graph_fname))
envcorr = graph_metrics.loc[:, :, 'envelope_correlation']
# load node metrics
node_metrics = xr.load_dataarray(os.path.join(xarray_dir, node_fname))


roi_degree = node_metrics.loc[:, :, 'degree', roi_names]
delta_roi_degree = roi_degree.loc['attend'] - roi_degree.loc['ignore']
node_predictors = delta_roi_degree.to_pandas()
beh_predictors = (beh.set_index('subj')
                     .filter(regex=r'^(vocab|secondary)', axis='columns'))

all_predictors = node_predictors.join(beh_predictors)

full_corrmat = pd.DataFrame(np.corrcoef(all_predictors, rowvar=False),
                            index=all_predictors.columns,
                            columns=all_predictors.columns)
corrmat = (full_corrmat
           .filter(node_predictors.columns, axis='index')
           .filter(beh_predictors.columns, axis='columns'))

fig, ax = plt.subplots()
sns.heatmap(corrmat, square=False, ax=ax, cmap='RdBu', vmin=-1, vmax=1)
ax.set_xticklabels(
    ['\n'.join(lab.split('_')) for lab in beh_predictors.columns],
    rotation=0)
fig.suptitle('Pearson correlations: Δ Degree (attend−ignore) vs vocab score',
             weight='bold', size='larger')
fig.set_size_inches(7, 7)
fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15)
fig.savefig(os.path.join(plot_dir, 'degree-vs-vocab.pdf'))
