#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot edge contributions to difference between conditions as a circular graph.

authors: Daniel McCloy
license: MIT
"""

import os
from functools import reduce

import numpy as np
import xarray as xr
import mne
from mne_connectivity.viz import plot_connectivity_circle

from f2f_helpers import load_paths, load_params, get_skip_regexp

# flags
freq_band = 'theta'
conditions = ['attend', 'ignore']
parcellation = 'hickok_corbetta'  # 'aparc'
threshold_prop = 0.15
n_sec = 7  # epoch duration to use

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
xarray_dir = os.path.join(results_dir, 'xarrays')
plot_dir = os.path.join(results_dir, 'figs', 'graph-metrics')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
roi_edge_dict = load_params(os.path.join(param_dir, 'roi-edges.yaml'))
roi_edges = roi_edge_dict[parcellation]
roi_nodes = sorted(set(reduce(tuple.__add__, roi_edges)))

# load all labels
labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml')
                             )[parcellation]
regexp = get_skip_regexp(labels_to_skip)
labels = mne.read_labels_from_annot(surrogate, parc=parcellation,
                                    regexp=regexp, subjects_dir=subjects_dir)
label_dict = {label.name: label for label in labels}

for use_edge_rois in (False, True):
    roi = 'roi' if use_edge_rois else 'all'
    # load xarrays
    slug = f'{parcellation}-{n_sec}sec-{freq_band}-band-{roi}-edges'
    fname = f'{slug}-lambda-hat.nc'
    lambda_hat = xr.load_dataarray(os.path.join(xarray_dir, fname))
    fname = f'{slug}-lambda-sq.nc'
    lambda_sq = xr.load_dataarray(os.path.join(xarray_dir, fname))

    # arrange labels in sensible order (by y coordinate of label centroid)
    label_names = lambda_sq.coords['region_1'].values.tolist()
    label_ypos = {name: np.mean(label_dict[name].pos[:, 1])
                  for name in label_names}

    lh_names = list(filter(lambda x: x.endswith('-lh'), label_names))
    lh_ypos_order = np.argsort([label_ypos[name] for name in lh_names])
    lh_names = np.array(lh_names)[lh_ypos_order].tolist()

    rh_names_stripped = [name.split('-')[0]
                         for name in set(label_names) - set(lh_names)]
    rh_names = list()
    for name in lh_names:
        stem = name.split('-')[0]
        if stem in rh_names_stripped:
            rh_names.append(f'{stem}-rh')
    leftovers = list(set(label_names) - set(lh_names) - set(rh_names))
    rh_ypos = [label_ypos[name] for name in rh_names]
    leftover_ypos = [label_ypos[name] for name in leftovers]
    indices = np.searchsorted(rh_ypos, leftover_ypos)
    for offset, ix in enumerate(np.argsort(indices)):
        rh_names.insert(indices[ix] + offset, leftovers[ix])
    node_order = lh_names + rh_names[::-1]

    # colors follows original order, not node order   ↓↓↓↓↓↓↓↓↓↓↓
    node_colors = [label_dict[name].color for name in label_names]

    # if aparc & using ROI, dim any colors that aren't in our ROI edge list
    if parcellation == 'aparc':
        for label in labels:
            if label.name not in roi_nodes:
                # make darker
                label.color = tuple(np.array([0.1, 0.1, 0.1, 1]) * label.color)

    # plot
    n_lines = lambda_hat.size
    extreme = np.abs(lambda_sq).max().values
    vlims = dict(vmin=-extreme, vmax=extreme)
    node_angles = mne.viz.circular_layout(
        label_names, node_order, start_pos=90,
        group_boundaries=[0, len(lh_names)])
    fig, ax = plot_connectivity_circle(
        lambda_sq.values, lambda_sq.coords['region_1'].values, n_lines=n_lines,
        node_angles=node_angles, node_colors=node_colors, title='',
        interactive=False, show=False, facecolor='w', textcolor='k',
        node_edgecolor='none', colormap='RdBu', colorbar_size=0.4,
        colorbar_pos=(1., 0.1), linewidth=1, **vlims)
    fig.set_size_inches(8, 8)
    fig.suptitle('Relative contribution to network connectivity,\n'
                 'difference between conditions (attend minus ignore)\n'
                 f'{parcellation} parcellation')
    fig.savefig(os.path.join(plot_dir, f'{slug}-connectivity-circle.pdf'))
