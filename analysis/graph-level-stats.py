#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute graph metrics.

authors: Daniel McCloy
license: MIT
"""

import os
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.components import is_connected
from networkx.linalg.laplacianmatrix import laplacian_matrix
from scipy.stats import chi2
from scipy.special import comb
from scipy.linalg import sqrtm
import seaborn as sns
import xarray as xr
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import mne
import mne_connectivity

from f2f_helpers import (load_paths, load_subjects, load_params, get_slug,
                         get_skip_regexp)

corpcor = importr('corpcor')
numpy2ri.activate()


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
sns.set(font_scale=0.8)
cluster_plot = False
use_edge_rois = True
figsize = (8, 8) if use_edge_rois else (32, 32)

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
epo_dir = os.path.join(results_root_dir, 'epochs')
param_dir = os.path.join('..', 'params')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
xarray_dir = os.path.join(results_dir, 'xarrays')
plot_dir = os.path.join(results_dir, 'figs', 'graph-metrics')
for _dir in (xarray_dir, plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
epoch_strategies = load_params(os.path.join(param_dir, 'min_epochs.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))
durations = [epoch_dict['length'] for epoch_dict in epoch_strategies]
roi_dict = load_params(os.path.join(param_dir, 'rois.yaml'))[parcellation]
roi_edges = load_params(os.path.join(param_dir, 'roi-edges.yaml'))
roi_nodes = reduce(tuple.__add__, roi_edges)

# load all labels
labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml')
                             )[parcellation]
regexp = get_skip_regexp(labels_to_skip)
labels = mne.read_labels_from_annot(surrogate, parc=parcellation,
                                    regexp=regexp, subjects_dir=subjects_dir)
label_names = [label.name for label in labels]
# load ROIs
roi_names = list()
for hemi, roi_name_dict in roi_dict.items():
    for roi_name in roi_name_dict:
        roi_names.append(f'{roi_name}-{hemi}')

# containers
results = dict()

for epoch_dict in epoch_strategies:
    n_sec = int(epoch_dict['length'])
    results[n_sec] = dict()
    # track which subjs are available at this epoch length
    this_excludes = {subj for subj in excludes if n_sec in excludes[subj]}
    this_subjects = sorted(set(subjects) - this_excludes)
    n_subj = len(this_subjects)
    # container for raw correlations, adjacencies, and graph laplacians
    _dtype = np.float64
    _min = np.finfo(_dtype).min
    measures = ['envelope_correlation', 'adjacency', 'graph_laplacian',
                'orthogonal_proj_matrix', 'lambda_squared']
    shape = (len(conditions), n_subj, len(measures), len(labels), len(labels))
    coords = dict(
        condition=conditions, subject=this_subjects, measure=measures,
        region_1=label_names, region_2=label_names)
    conn_measures = xr.DataArray(np.full(shape, fill_value=_min, dtype=_dtype),
                                 coords=coords,
                                 name='graph-level connectivity measures')
    # loop over conditions & subjects
    for condition in conditions:
        for subj in this_subjects:
            # load connectivity
            slug = get_slug(subj, freq_band, condition, parcellation)
            conn_fname = (f'{slug}-{n_sec}sec-envelope-correlation.nc')
            conn_fpath = os.path.join(conn_dir, conn_fname)
            conn = mne_connectivity.read_connectivity(conn_fpath)
            # strip "f2f-" off of the label names in the connectivity object
            if parcellation == 'f2f_custom':
                conn.attrs['node_names'] = np.array([name[4:]
                                                     for name in conn.names])
            # envelope correlations
            conn_matrix = conn.get_data('dense').squeeze()
            conn_measures.loc[condition, subj,
                              'envelope_correlation'] = conn_matrix
            # adjacency
            n_conn = len(conn.names) * (len(conn.names) - 1) / 2
            n_keep = np.ceil(n_conn * threshold_prop).astype(int)
            quantile = 1 - threshold_prop
            indices = np.tril_indices_from(conn_matrix, k=-1)
            threshold = np.quantile(conn_matrix[indices], quantile)
            adjacency = (conn_matrix > 0).astype(int)
            conn_measures.loc[condition, subj, 'adjacency'] = adjacency
            # graph laplacian
            graph = nx.Graph(
                conn_measures.loc[condition, subj, 'adjacency'].to_pandas())
            graph.clear_edges()
            mask = np.logical_and(
                adjacency.astype(bool),
                np.tril(np.ones_like(adjacency), k=-1).astype(bool))
            edges = [(list(graph)[ix], list(graph)[iy], w) for ix, iy, w in
                     zip(*np.nonzero(mask), conn_matrix[mask])]
            graph.add_weighted_edges_from(edges)
            assert is_connected(graph)  # assumption of method
            laplacian = laplacian_matrix(graph)
            conn_measures.loc[condition, subj,
                              'graph_laplacian'] = laplacian.toarray()
    # placeholder, used later only in aggregate (not for each subj)
    conn_measures.loc[:, :, 'lambda_squared'] = 0
    # prep for restricting to specific edge ROIs
    conn_measures.loc[:, :, 'orthogonal_proj_matrix'] = 0
    for _node1, _node2 in roi_edges:
        conn_measures.loc[:, :, 'orthogonal_proj_matrix', _node1, _node2] = 1
        conn_measures.loc[:, :, 'orthogonal_proj_matrix', _node2, _node1] = 1
    # make sure every cell got filled
    assert np.all(conn_measures > _min)
    results[n_sec]['xarray'] = conn_measures
    roi = 'roi' if use_edge_rois else 'all'
    fname = (f'{parcellation}-{n_sec}sec-{freq_band}-band-{roi}-edges.nc')
    conn_measures.to_netcdf(os.path.join(xarray_dir, fname))

    # compute mean laplacians
    mean_over_subj = conn_measures.mean(dim='subject')
    mean_laplacians = mean_over_subj.loc[:, 'graph_laplacian']
    # absence of off-diagonal structural zeros is an assumption of the method
    for _ml in mean_laplacians:
        offdiag = _ml.values[np.tril_indices_from(_ml, k=-1)]
        assert np.all(offdiag != 0)
    results[n_sec]['means'] = mean_over_subj
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
    # save some results
    results[n_sec]['sigma_inv'] = sigma_inv
    results[n_sec]['t_obs'] = t_observed
    results[n_sec]['pval'] = pval
    # estimate contribution of each edge to the difference between conditions
    inv_sqrt_sigma = sqrtm(sigma_inv)
    lambda_hat = inv_sqrt_sigma @ mean_diff_halfvec
    np.testing.assert_almost_equal(t_observed,
                                   n_subj * np.sum(lambda_hat ** 2))
    results[n_sec]['lambda_hat'] = lambda_hat
    # compute relative contribution of each edge to the difference
    lambda_sq = (mean_over_subj.loc[:, 'lambda_squared']
                               .mean(dim='condition'))
    triu_indices = np.triu_indices_from(lambda_sq)
    if use_edge_rois:
        triu_indices = tuple(ixs[halfvec_mask] for ixs in triu_indices)
    lambda_sq.values[triu_indices] = lambda_hat ** 2
    lambda_sq.values.T[triu_indices] = lambda_hat ** 2
    # make sure we got the right ones
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
    _slice = slice(None, thresh_ix, -1)
    for _nodes, _hat in zip(ranked_edges[_slice], ranked_lambda_hat[_slice]):
        contrib = np.sign(_hat) * (_hat ** 2)
        print(f'{_nodes[0]:27} ←→ {_nodes[1]:>27}'
              f'    {round(contrib, 3):<+06.03}')
    # sort rows/cols by hemisphere
    df = lambda_sq.to_pandas()
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
    roi = 'roi-' if use_edge_rois else ''
    fname = (f'{parcellation}-{n_sec}sec-{freq_band}-band-'
             f'{roi}edge-contributions-attend_minus_ignore{clust}.pdf')
    if cluster_plot:
        cg = sns.clustermap(df, figsize=figsize)
        fig = cg.fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df, square=True, ax=ax)
    fig.savefig(os.path.join(plot_dir, fname))
    del fig
