#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute and plot graph-level metrics and stats.

authors: Daniel McCloy
license: MIT
"""

from functools import reduce
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import comb
from scipy.linalg import sqrtm
import seaborn as sns
import xarray as xr
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from f2f_helpers import load_paths, load_subjects, load_params


def get_halfvec(array, k=0):
    indices = np.triu_indices_from(array, k=k)
    halfvec = (array.values[indices] if isinstance(array, xr.DataArray) else
               array[indices])
    return halfvec


# flags
freq_band = 'theta'
conditions = ['attend', 'ignore']
parcellation = 'f2f_custom'  # 'aparc'
threshold_prop = 0.15
n_sec = 7  # epoch duration to use
sns.set(font_scale=0.8)
cluster_plot = False

# enable interface to R
corpcor = importr('corpcor')
numpy2ri.activate()

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
epo_dir = os.path.join(results_root_dir, 'epochs')
param_dir = os.path.join('..', 'params')
xarray_dir = os.path.join(results_dir, 'xarrays')
stats_dir = os.path.join(results_dir, 'stats')
plot_dir = os.path.join(results_dir, 'figs', 'graph-metrics')
for _dir in (stats_dir, plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))
roi_edges = load_params(os.path.join(param_dir, 'roi-edges.yaml'))
roi_nodes = reduce(tuple.__add__, roi_edges)

# how many subjs are available at this epoch length?
this_excludes = {subj for subj in excludes if n_sec in excludes[subj]}
this_subjects = sorted(set(subjects) - this_excludes)
n_subj = len(this_subjects)

# load xarray
fname = f'{parcellation}-{n_sec}sec-{freq_band}-band.nc'
conn_measures = xr.open_dataarray(os.path.join(xarray_dir, fname))

for use_edge_rois in (False, True):
    # compute mean laplacians
    mean_over_subj = conn_measures.mean(dim='subject')
    mean_laplacians = mean_over_subj.loc[:, 'graph_laplacian']
    # absence of off-diagonal structural zeros is an assumption of the method
    for _ml in mean_laplacians:
        offdiag = _ml.values[np.tril_indices_from(_ml, k=-1)]
        assert np.all(offdiag != 0)
    # reduce matrices to half-vectorized form (removes redundant entries)
    mean_halfvecs = np.array([get_halfvec(_ml) for _ml in mean_laplacians])
    mean_diff_halfvec = np.diff(mean_halfvecs, axis=0).squeeze()
    subj_laplacians = conn_measures.loc[:, :, 'graph_laplacian']
    subj_diffs = np.diff(subj_laplacians, axis=0).squeeze()
    subj_diff_halfvecs = np.array([get_halfvec(arr) for arr in subj_diffs])
    # restrict halfvecs to just the ROI edges (if desired)
    halfvec_mask = np.ones_like(mean_diff_halfvec).astype(bool)
    if use_edge_rois:
        halfvec_mask = get_halfvec(
            mean_over_subj.loc['attend', 'orthogonal_proj_matrix']
            ).astype(bool)
        assert halfvec_mask.sum() == len(roi_edges)
    subj_diff_halfvecs = subj_diff_halfvecs[:, halfvec_mask]
    mean_diff_halfvec = mean_diff_halfvec[halfvec_mask]
    # call out to R to use special shrinkage methods designed for graphs
    # use corpcor.cov_shrink(...) if we need sigma (not just its inverse)
    sigma_inv = corpcor.invcov_shrink(subj_diff_halfvecs)
    # calculate one-sample t-statistic
    t_observed = n_subj * (mean_diff_halfvec @ sigma_inv @ mean_diff_halfvec)
    degrees_of_freedom = comb(len(mean_diff_halfvec), 2, exact=True)
    pval = chi2.cdf(t_observed, df=degrees_of_freedom)
    # estimate contribution of each edge to the difference between conditions
    sqrt_inv_sigma = sqrtm(sigma_inv)
    lambda_hat = sqrt_inv_sigma @ mean_diff_halfvec
    np.testing.assert_almost_equal(t_observed,
                                   n_subj * np.sum(lambda_hat ** 2))
    lambda_sq = mean_over_subj.loc[:, 'lambda_squared'].mean(dim='condition')
    triu_indices = np.triu_indices_from(lambda_sq)
    if use_edge_rois:
        triu_indices = tuple(ixs[halfvec_mask] for ixs in triu_indices)
    lambda_sq.values[triu_indices] = np.sign(lambda_hat) * lambda_hat ** 2
    lambda_sq.values.T[triu_indices] = np.sign(lambda_hat) * lambda_hat ** 2
    # make sure we got the right indices
    if use_edge_rois:
        for dim, region in enumerate(lambda_sq.coords.dims):
            nodes = lambda_sq.coords[region].values[np.nonzero(lambda_sq)[dim]]
            assert set(nodes) == set(roi_nodes)
    # sort edges by strength of contribution (increasing)
    ranked_indices = np.argsort(lambda_hat ** 2)
    ranked_lambda_hat = lambda_hat[ranked_indices]
    ranked_edge_indices = tuple(x[ranked_indices] for x in triu_indices)
    ranked_edges = list(zip(
        *(lambda_sq[ranked_edge_indices].coords[reg].values.tolist()
          for reg in lambda_sq.coords.dims)))
    # set cutoff for which ones to print
    q = 75 if use_edge_rois else 99
    abs_thresh = np.percentile(lambda_hat ** 2, q)
    thresh_ix = np.searchsorted(ranked_lambda_hat ** 2, abs_thresh)
    # print which edges change the most
    print(f'TOP {100 - q}% EDGES WITH STRONGEST DIFFERENCE BETWEEN '
          'CONDITIONS (ATTEND MINUS IGNORE):')
    _slice = slice(None, thresh_ix, -1)  # threshold; ascending → descending
    for _nodes, _hat in zip(ranked_edges[_slice], ranked_lambda_hat[_slice]):
        contrib = np.sign(_hat) * (_hat ** 2)
        print(f'{_nodes[0]:27} ←→ {_nodes[1]:>27}'
              f'    {round(contrib, 3):<+06.03}')

    # save stats info to file
    edge_contributions = {
        f'{" <--> ".join(_edge)}': float(_hat)
        for _edge, _hat in zip(ranked_edges[::-1], ranked_lambda_hat[::-1])
    }
    output = dict(
        t_observed=float(t_observed),
        degrees_of_freedom=degrees_of_freedom,
        p_value=float(pval),
        lambda_hat=edge_contributions
    )
    roi = 'roi' if use_edge_rois else 'all'
    fname = f'{parcellation}-{n_sec}sec-{freq_band}-band-{roi}-edges.yaml'
    with open(os.path.join(stats_dir, fname), 'w') as f:
        yaml.dump(output, f)
    # save lambda hat vector and lambda squared matrix
    coords = {f'region_{ix + 1}': ('edge', list(region))
              for ix, region in enumerate(zip(*ranked_edges))}
    lambda_hat_xarray = xr.DataArray(ranked_lambda_hat,
                                     coords=coords,
                                     dims=['edge'],
                                     name='lambda hat')
    slug = f'{parcellation}-{n_sec}sec-{freq_band}-band-{roi}-edges'
    fname = f'{slug}-lambda-hat.nc'
    lambda_hat_xarray.to_netcdf(os.path.join(xarray_dir, fname))
    fname = f'{slug}-lambda-sq.nc'
    lambda_sq.to_netcdf(os.path.join(xarray_dir, fname))

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
    fname = (f'{parcellation}-{n_sec}sec-{freq_band}-band-'
             f'{roi}-edges-attend_minus_ignore{clust}.pdf')
    figsize = (8, 8) if use_edge_rois else (32, 32)
    if cluster_plot:
        cg = sns.clustermap(df, figsize=figsize)
        fig = cg.fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df, square=True, ax=ax)
    fig.savefig(os.path.join(plot_dir, fname))
    del fig
