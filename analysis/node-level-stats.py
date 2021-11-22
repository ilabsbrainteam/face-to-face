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
# XXX from networkx.algorithms.smallworld import omega, sigma
# XXX from networkx.algorithms.shortest_paths.generic import average_shortest_path_length  # noqa E501
import pandas as pd
from scipy.stats import ttest_rel
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
results = dict()

for epoch_dict in epoch_strategies:
    n_sec = int(epoch_dict['length'])
    results[n_sec] = dict()
    # track which subjs are available at this epoch length
    this_excludes = {subj for subj in excludes if n_sec in excludes[subj]}
    this_subjects = sorted(set(subjects) - this_excludes)
    n_subj = len(this_subjects)
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
            # load connectivity
            slug = get_slug(subj, freq_band, condition, parcellation)
            conn_fname = (f'{slug}-{n_sec}sec-envelope-correlation.nc')
            conn_fpath = os.path.join(conn_dir, conn_fname)
            conn = mne_connectivity.read_connectivity(conn_fpath)
            # strip "f2f-" off of the label names in the connectivity object
            if parcellation == 'f2f_custom':
                conn.attrs['node_names'] = np.array([name[4:]
                                                     for name in conn.names])
            # load envelope correlations and make graph
            conn_matrix = conn.get_data('dense').squeeze()
            df = pd.DataFrame(
                conn_matrix, index=conn.names, columns=conn.names)
            graph = nx.Graph(df)
            # node metrics
            clust_coef = nx.clustering(graph)
            betw_cent = betweenness_centrality(graph)
            degree = mne_connectivity.degree(conn, threshold_prop)
            # fill in the xarray
            for this_label, this_degree in zip(conn.names, degree):
                this_metrics = np.array([
                    this_degree, clust_coef[this_label], betw_cent[this_label]
                ])
                metrics.loc[condition, subj, :, this_label] = this_metrics
    # make sure every cell got filled
    assert np.all(metrics >= 0)

    # node-level stats
    stats = xr.DataArray(list(ttest_rel(*metrics, axis=0)),
                         coords=dict(stat=['tval', 'pval'],
                         metric=list(node_metrics),
                         region=label_names))
    for metric in node_metrics:
        results[n_sec][metric] = dict()
        this_stats = stats.loc[:, metric]
        signifs = this_stats.where(this_stats.loc['pval'] < 0.05, drop=True)
        signif_regions = signifs.coords['region'].values.tolist()
        attend_larger_than_ignore = this_stats.where(
            np.logical_and(this_stats.loc['pval'] < 0.05,
                           this_stats.loc['tval'] > 0), drop=True)
        # Šidák correction
        sidak = 1 - (0.95) ** (1 / len(roi_names))  # or label_names if no ROIs
        signif_regions_sidak = this_stats.where(
            this_stats.loc['pval'] < sidak, drop=True
            ).coords['region'].values.tolist()
        # store results
        for name, result in dict(
                stats=stats,
                signifs=signifs,
                signif_regions=signif_regions,
                attend_larger_than_ignore=attend_larger_than_ignore,
                signif_regions_sidak=signif_regions_sidak
                ).items():
            results[n_sec][metric][name] = result

        for kind, _signif in dict(uncorrected=signif_regions,
                                  corrected=signif_regions_sidak).items():
            # plot signifs
            brain = Brain(
                surrogate, hemi='split', surf='inflated', size=(1200, 900),
                cortex='low_contrast', views=['lateral', 'medial'],
                background='white', subjects_dir=subjects_dir)
            regexp = '|'.join(_signif)
            # avoid empty regexp loading all labels
            signif_labels = (
                list() if not len(_signif) else
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
            fname = (f'{parcellation}-{n_sec}sec-{metric}-{kind}-signif-'
                     'labels.png')
            brain.save_image(os.path.join(plot_dir, fname))
            brain.close()
            del brain
