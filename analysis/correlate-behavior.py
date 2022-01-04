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
import scipy.stats as ss
import seaborn as sns
import xarray as xr

from f2f_helpers import load_paths, load_params

# flags
n_sec = 7  # epoch duration to use

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
xarray_dir = os.path.join(results_dir, 'xarrays')
beh_dir = os.path.join('..', 'behavioral')
plot_dir = os.path.join(results_dir, 'figs', 'behavior-correlations')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load params
all_roi_dicts = load_params(os.path.join(param_dir, 'rois.yaml'))
analysis_bands = load_params(os.path.join(param_dir, 'analysis_bands.yaml'))

# load behavioral data
beh = pd.read_csv(os.path.join(beh_dir, 'f2f-behavioral-data.csv'))

for parcellation, roi_dict in all_roi_dicts.items():
    # load ROIs
    roi_names = list()
    for hemi, roi_name_dict in roi_dict.items():
        for roi_name in roi_name_dict:
            roi_names.append(f'{roi_name}-{hemi}')

    # loop over analysis bands
    for freq_band in analysis_bands:

        # xarray filenames
        slug = f'{parcellation}-{n_sec}sec-{freq_band}-band'
        graph_fname = f'{slug}-graph-metrics.nc'
        node_fname = f'{slug}-all-edges-node-metrics.nc'

        # load graph metrics
        graph_metrics = xr.load_dataarray(os.path.join(xarray_dir,
                                                       graph_fname))
        envcorr = graph_metrics.loc[:, :, 'envelope_correlation']

        # load node metrics
        node_metrics = xr.load_dataarray(os.path.join(xarray_dir, node_fname))
        roi_degree = node_metrics.loc[:, :, 'degree', roi_names]
        delta_roi_degree = roi_degree.loc['attend'] - roi_degree.loc['ignore']

        # prep predictors
        node_predictors = delta_roi_degree.to_pandas()
        beh_predictors = (beh.set_index('subj')
                             .filter(items=node_predictors.index, axis='index')
                             .filter(regex=r'^(vocab|secondary)',
                                     axis='columns'))
        all_predictors = node_predictors.join(beh_predictors)
        n = all_predictors.shape[0]
        # compute correlations and p-values
        full_corrmat = all_predictors.corr('pearson')
        corrmat = (full_corrmat.filter(node_predictors.columns, axis='index')
                               .filter(beh_predictors.columns, axis='columns'))
        # next 2 lines adapted from docstring of ss.pearsonr()
        distribution = ss.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
        pvals = pd.DataFrame(2 * distribution.cdf(-np.abs(corrmat)),
                             index=corrmat.index,
                             columns=corrmat.columns)
        sidak_thresh = 1 - (0.95) ** (1 / corrmat.size)
        pval_text = pvals.applymap(
            lambda x: f'p={x:.2f}{" *" if x <= sidak_thresh else ""}')
        annot = corrmat.applymap(lambda x: f'r={x:.2f}\n') + pval_text
        annot = annot.applymap(lambda x: x.replace('-', '−'))  # typography

        fig, ax = plt.subplots()
        sns.heatmap(
            corrmat, square=False, ax=ax, annot=annot, fmt='',
            annot_kws=dict(size='smaller'), cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticklabels(
            ['\n'.join(lab.split('_')) for lab in beh_predictors.columns],
            rotation=0)
        parc = (parcellation.upper() if parcellation == 'f2f' else
                parcellation.capitalize())
        band = freq_band.replace('_', ' ')
        title = ('Pearson correlations: Δ Degree (attend−ignore) vs '
                 f'behavioral measures\n{parc} parcellation, {band} band')
        fig.suptitle(title, size='larger')
        fig.set_size_inches(7, 7)
        fig.subplots_adjust(left=0.25, right=0.95, bottom=0.15)
        fname = f'{parcellation}-{freq_band}_band-degree-vs-vocab.pdf'
        fig.savefig(os.path.join(plot_dir, fname))
