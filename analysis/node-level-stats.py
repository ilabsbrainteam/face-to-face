#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute and plot node-level metrics and stats.

authors: Daniel McCloy
license: MIT
"""

import os

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

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
epo_dir = os.path.join(results_root_dir, 'epochs')
param_dir = os.path.join('..', 'params')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
xarray_dir = os.path.join(results_dir, 'xarrays')
plot_dir = os.path.join(results_dir, 'figs', 'node-metrics')
for _dir in (plot_dir,):
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
fname = f'{parcellation}-{n_sec}sec-{freq_band}-band.nc'
conn_measures = xr.open_dataarray(os.path.join(xarray_dir, fname))

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
# for plotting
medial_wall_labels = {parc: (
        'cuneus', 'precuneus', 'paracentral', 'superiorfrontal',
        'medialorbitofrontal', 'rostralanteriorcingulate',
        'caudalanteriorcingulate', 'posteriorcingulate', 'isthmuscingulate',
        'parahippocampal', 'entorhinal', 'fusiform', 'lingual',
        'pericalcarine')
    for parc in ('aparc', 'aparc_sub', 'f2f_custom')}

# containers
Brain = mne.viz.get_brain_class()

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
        graph = nx.Graph(conn_measures
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
results = dict()
metric_arrays = dict(roi=roi_metrics, all=metrics)

# node-level stats
for scope, _xarray in metric_arrays.items():
    results[scope] = dict()
    sidak = 1 - (0.95) ** (1 / _xarray.shape[-1])  # Šidák correction
    stats = xr.DataArray(list(ttest_rel(*_xarray, axis=0)),
                         coords=dict(stat=['tval', 'pval'],
                                     metric=_xarray.coords['metric'],
                                     region=_xarray.coords['region']))
    for metric in _xarray.coords['metric'].values:
        results[scope][metric] = dict()
        for thresh_kind, thresh in dict(sidak=sidak, uncorrected=0.05).items():
            results[scope][metric][thresh_kind] = dict()
            this_stats = stats.loc['pval', metric]
            signif = this_stats.where(this_stats < thresh, drop=True)
            signif_regions = signif.coords['region'].values.tolist()
            attend_larger_than_ignore = this_stats.where(
                np.logical_and(this_stats < thresh, this_stats > 0), drop=True)
            results[scope][metric][thresh_kind]['signif'] = signif
            results[scope][metric][thresh_kind]['regions'] = signif_regions
            results[scope][metric][thresh_kind]['attend>ignore'] = (
                attend_larger_than_ignore)

for scope in results:
    for metric in results[scope]:
        for thresh_kind in results[scope][metric]:
            regions = results[scope][metric][thresh_kind]['regions']
            # plot signifs
            brain = Brain(
                surrogate, hemi='split', surf='inflated', size=(1200, 900),
                cortex='low_contrast', views=['lateral', 'medial'],
                background='white', subjects_dir=subjects_dir)
            regexp = '|'.join(regions)
            # avoid empty regexp loading all labels
            signif_labels = (
                list() if not len(regions) else
                mne.read_labels_from_annot(
                    surrogate, parcellation, regexp=regexp,
                    subjects_dir=subjects_dir)
                )
            # prevent text overlap
            text_bookkeeping = {(row, col): list() for row in (0, 1)
                                for col in (0, 1)}
            # draw labels and add label names
            for label in signif_labels:
                brain.add_label(label, alpha=0.5)
                brain.add_label(label, borders=True)
                col = int(label.hemi == 'rh')
                row = int(label.name.rsplit('-')[0].rsplit('_')[0]
                          in medial_wall_labels[parcellation])
                y = 0.02 + 0.06 * len(text_bookkeeping[(row, col)])
                text_bookkeeping[(row, col)].append(label.name)
                brain.add_text(
                    0.05, y, text=label.name.rsplit('-')[0], name=label.name,
                    row=row, col=col, color=label.color, font_size=12)
            fname = (f'{parcellation}-{n_sec}sec-{metric}-'
                     f'{thresh_kind}-signif-{scope}-labels.png')
            brain.save_image(os.path.join(plot_dir, fname))
            brain.close()
            del brain
