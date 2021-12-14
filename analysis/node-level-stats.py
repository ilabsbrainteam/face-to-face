#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute node-level stats.

authors: Daniel McCloy
license: MIT
"""

import os
import yaml

import numpy as np
import networkx as nx
from networkx.algorithms.centrality import betweenness_centrality
from scipy.stats import ttest_rel
import xarray as xr
import mne

from f2f_helpers import load_paths, load_subjects, load_params, get_skip_regexp

# flags
freq_band = 'theta'
conditions = ['attend', 'ignore']
parcellation = 'f2f_custom'  # 'aparc'
threshold_prop = 0.15
n_sec = 7  # epoch duration to use
plot = False

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
epo_dir = os.path.join(results_root_dir, 'epochs')
param_dir = os.path.join('..', 'params')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
xarray_dir = os.path.join(results_dir, 'xarrays')
stats_dir = os.path.join(results_dir, 'stats')
plot_dir = os.path.join(results_dir, 'figs', 'node-metrics')
for _dir in (stats_dir, plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
epoch_strategies = load_params(os.path.join(param_dir, 'min_epochs.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))
durations = [epoch_dict['length'] for epoch_dict in epoch_strategies]
roi_dict = load_params(os.path.join(param_dir, 'rois.yaml'))[parcellation]

# how many subjs are available at this epoch length?
this_excludes = {subj for subj in excludes if n_sec in excludes[subj]}
this_subjects = sorted(set(subjects) - this_excludes)
n_subj = len(this_subjects)

# load xarray
slug = f'{parcellation}-{n_sec}sec-{freq_band}-band'
fname = f'{slug}-graph-metrics.nc'
conn_metrics = xr.load_dataarray(os.path.join(xarray_dir, fname))

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

# container for node-level metrics
node_metrics = ['degree',
                'clustering_coefficient',
                'betweenness_centrality']
shape = (len(conditions), n_subj, len(node_metrics), len(labels))
metrics = xr.DataArray(np.full(shape, fill_value=-1, dtype=float),
                       coords=dict(condition=conditions,
                                   subject=this_subjects,
                                   metric=node_metrics,
                                   region=label_names),
                       name='node-level connectivity metrics')

# loop over conditions & subjects
for condition in conditions:
    for subj in this_subjects:
        # make graph
        graph = nx.Graph(conn_metrics
                         .loc[condition, subj,
                              'thresholded_weighted_adjacency']
                         .to_pandas())
        # node metrics
        node_metrics = dict(
            degree=graph.degree,
            clustering_coefficient=nx.clustering(graph),
            betweenness_centrality=betweenness_centrality(graph),
        )
        # fill in xarray
        for metric, result in node_metrics.items():
            for region, value in dict(result).items():
                metrics.loc[condition, subj, metric, region] = value

# make sure every cell got filled
assert np.all(metrics >= 0)

# subset to ROIs only
roi_metrics = metrics.loc[..., np.array(roi_names)]

# container
metric_arrays = dict(roi=roi_metrics, all=metrics)

# node-level stats
for scope, _xarray in metric_arrays.items():
    output = dict()
    sidak = 1 - (0.95) ** (1 / _xarray.shape[-1])  # Šidák correction
    stats = xr.DataArray(list(ttest_rel(*_xarray, axis=0)),
                         coords=dict(stat=['tval', 'pval'],
                                     metric=_xarray.coords['metric'],
                                     region=_xarray.coords['region']))
    for thresh_kind, thresh in dict(sidak=sidak, uncorrected=0.05).items():
        output[thresh_kind] = dict()
        for metric in _xarray.coords['metric'].values:
            output[thresh_kind][str(metric)] = dict()
            out_stats = stats.loc[:, metric].where(
                stats.loc['pval', metric] < thresh, drop=True)
            for reg, tval, pval in zip(out_stats['region'].values.tolist(),
                                       out_stats.loc['tval'].values.tolist(),
                                       out_stats.loc['pval'].values.tolist()):
                output[thresh_kind][str(metric)][reg] = dict(
                    tval=float(tval), pval=float(pval))
    # save xarray
    fname = f'{slug}-{scope}-edges-node-metrics.nc'
    _xarray.to_netcdf(os.path.join(xarray_dir, fname))
    # save stats
    fname = f'{slug}-{scope}-edges-node-level-stats.yaml'
    with open(os.path.join(stats_dir, fname), 'w') as f:
        yaml.dump(output, f)
