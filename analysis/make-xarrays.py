#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare data for analysis of graph metrics.

authors: Daniel McCloy
license: MIT
"""

import os

import numpy as np
import networkx as nx
from networkx.algorithms.components import is_connected
from networkx.linalg.laplacianmatrix import laplacian_matrix
import xarray as xr
import mne
import mne_connectivity

from f2f_helpers import (load_paths, load_subjects, load_params, get_slug,
                         get_skip_regexp)

# flags
freq_band = 'theta'
conditions = ['attend', 'ignore']
threshold_prop = 0.15

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
epo_dir = os.path.join(results_root_dir, 'epochs')
param_dir = os.path.join('..', 'params')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
xarray_dir = os.path.join(results_dir, 'xarrays')
for _dir in (xarray_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
epoch_strategies = load_params(os.path.join(param_dir, 'min_epochs.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))
roi_edge_dict = load_params(os.path.join(param_dir, 'roi-edges.yaml'))

for parcellation, roi_edges in roi_edge_dict.items():
    # load all labels
    labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml')
                                 )[parcellation]
    regexp = get_skip_regexp(labels_to_skip)
    labels = mne.read_labels_from_annot(
        surrogate, parc=parcellation, regexp=regexp, subjects_dir=subjects_dir)
    label_names = [label.name for label in labels]

    for epoch_dict in epoch_strategies:
        n_sec = int(epoch_dict['length'])
        # track which subjs are available at this epoch length
        this_excludes = {subj for subj in excludes if n_sec in excludes[subj]}
        this_subjects = sorted(set(subjects) - this_excludes)
        n_subj = len(this_subjects)
        # container for raw correlations, adjacencies, and graph laplacians
        _dtype = np.float64
        _min = np.finfo(_dtype).min
        metrics = ['envelope_correlation', 'adjacency',
                   'thresholded_weighted_adjacency',
                   'unthresholded_weighted_adjacency',
                   'graph_laplacian', 'orthog_proj_mat', 'lambda_squared']
        shape = (len(conditions), n_subj, len(metrics),
                 len(labels), len(labels))
        coords = dict(
            condition=conditions, subject=this_subjects, metric=metrics,
            region_1=label_names, region_2=label_names)
        graph_metrics = xr.DataArray(
            np.full(shape, fill_value=_min, dtype=_dtype),
            coords=coords,
            name='graph-level connectivity metrics')
        # loop over conditions & subjects
        for condition in conditions:
            for subj in this_subjects:
                # load connectivity
                slug = get_slug(subj, freq_band, condition, parcellation)
                conn_fname = (f'{slug}-{n_sec}sec-envelope-correlation.nc')
                conn_fpath = os.path.join(conn_dir, conn_fname)
                conn = mne_connectivity.read_connectivity(conn_fpath)
                # envelope correlations
                conn_matrix = conn.get_data('dense').squeeze()
                graph_metrics.loc[condition, subj,
                                  'envelope_correlation'] = conn_matrix
                # adjacency (thresholded)
                quantile = 1 - threshold_prop
                indices = np.tril_indices_from(conn_matrix, k=-1)
                threshold = np.quantile(conn_matrix[indices], quantile)
                adjacency = (conn_matrix > threshold).astype(int)
                graph_metrics.loc[condition, subj, 'adjacency'] = adjacency
                # weighted "adjacency" (thresholded and unthresholded)
                graph_metrics.loc[
                    condition, subj, 'unthresholded_weighted_adjacency'
                    ] = np.where(conn_matrix > 0, conn_matrix, adjacency)
                graph_metrics.loc[
                    condition, subj, 'thresholded_weighted_adjacency'
                    ] = np.where(conn_matrix > threshold, conn_matrix,
                                 adjacency)
                # graph laplacian
                graph = nx.Graph(graph_metrics
                                 .loc[condition, subj,
                                      'unthresholded_weighted_adjacency']
                                 .to_pandas())
                assert is_connected(graph)  # assumption of method
                laplacian = laplacian_matrix(graph)
                graph_metrics.loc[condition, subj,
                                  'graph_laplacian'] = laplacian.toarray()
        # placeholder, used later only in aggregate (not for each subj)
        graph_metrics.loc[:, :, 'lambda_squared'] = 0
        # prep for restricting to specific edge ROIs
        graph_metrics.loc[:, :, 'orthog_proj_mat'] = 0
        for _node1, _node2 in roi_edges:
            graph_metrics.loc[:, :, 'orthog_proj_mat', _node1, _node2] = 1
            graph_metrics.loc[:, :, 'orthog_proj_mat', _node2, _node1] = 1
        # make sure every cell got filled
        assert np.all(graph_metrics > _min)
        fname = f'{parcellation}-{n_sec}sec-{freq_band}-band-graph-metrics.nc'
        graph_metrics.to_netcdf(os.path.join(xarray_dir, fname))
