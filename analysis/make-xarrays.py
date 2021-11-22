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
parcellation = 'f2f_custom'  # 'aparc'
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
roi_edges = load_params(os.path.join(param_dir, 'roi-edges.yaml'))
roi_nodes = reduce(tuple.__add__, roi_edges)

# load all labels
labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml')
                             )[parcellation]
regexp = get_skip_regexp(labels_to_skip)
labels = mne.read_labels_from_annot(surrogate, parc=parcellation,
                                    regexp=regexp, subjects_dir=subjects_dir)
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
    fname = f'{parcellation}-{n_sec}sec-{freq_band}-band.nc'
    conn_measures.to_netcdf(os.path.join(xarray_dir, fname))
