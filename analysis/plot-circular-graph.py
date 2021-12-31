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
import matplotlib.pyplot as plt
import xarray as xr
import mne
from mne_connectivity.viz import plot_connectivity_circle

from f2f_helpers import load_paths, load_params, get_skip_regexp, get_halfvec

# flags
n_sec = 7  # epoch duration to use
strongest_edges = 0.01  # proportion of strongest edges to show

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
analysis_bands = load_params(os.path.join(param_dir, 'analysis_bands.yaml'))

for parcellation, roi_edges in roi_edge_dict.items():
    roi_nodes = sorted(set(reduce(tuple.__add__, roi_edges)))
    # load all labels
    labels_to_skip = load_params(os.path.join(param_dir, 'skip_labels.yaml')
                                 )[parcellation]
    regexp = get_skip_regexp(labels_to_skip)
    labels = mne.read_labels_from_annot(
        surrogate, parc=parcellation, regexp=regexp, subjects_dir=subjects_dir)
    label_dict = {label.name: label for label in labels}

    for freq_band in analysis_bands:
        for cond_name in ('all_conds', 'attend_vs_ignore'):
            for use_edge_rois in (False, True):
                roi = 'roi' if use_edge_rois else 'all'
                # load xarrays
                slug = (f'{cond_name}-{parcellation}-{n_sec}sec-'
                        f'{freq_band}-band-{roi}-edges')
                fname = f'{slug}-lambda-hat.nc'
                lambda_hat = xr.load_dataarray(os.path.join(xarray_dir, fname))
                fname = f'{slug}-lambda-sq.nc'
                lambda_sq = xr.load_dataarray(os.path.join(xarray_dir, fname))
                # arrange labels in sensible order (by y coordinate of label
                # centroid)
                label_names = lambda_sq.coords['region_1'].values.tolist()
                label_ypos = {name: np.mean(label_dict[name].pos[:, 1])
                              for name in label_names}

                lh_names = list(
                    filter(lambda x: x.endswith('-lh'), label_names))
                lh_ypos_order = np.argsort(
                    [label_ypos[name] for name in lh_names])
                lh_names = np.array(lh_names)[lh_ypos_order].tolist()
                # handle when LH and RH labels don't have same set of names
                rh_names_stripped = [
                    name.split('-')[0]
                    for name in set(label_names) - set(lh_names)]
                rh_names = list()
                for name in lh_names:
                    stem = name.split('-')[0]
                    if stem in rh_names_stripped:
                        rh_names.append(f'{stem}-rh')
                leftovers = list(
                    set(label_names) - set(lh_names) - set(rh_names))
                rh_ypos = [label_ypos[name] for name in rh_names]
                leftover_ypos = [label_ypos[name] for name in leftovers]
                indices = np.searchsorted(rh_ypos, leftover_ypos)
                # this next line *looks* incorrect (typically the enumeration
                # counter is called "ix", not the enumerated variable) but it's
                # not a mistake. Here, because we're inserting entries into a
                # list, our list grows with each iteration so we use the
                # enumeration index as an *offset* for our insertion index.
                for offset, ix in enumerate(np.argsort(indices)):
                    rh_names.insert(indices[ix] + offset, leftovers[ix])
                node_order = lh_names + rh_names[::-1]

                # colors follow original order, not node order    ↓↓↓↓↓↓↓↓↓↓↓
                node_colors = [label_dict[name].color for name in label_names]

                # plot
                node_angles = mne.viz.circular_layout(
                    label_names, node_order, start_pos=90,
                    group_boundaries=[0, len(lh_names)])
                # be meticulous about our colormap limits. This ignores the
                # degree values for each node (along the diagonal of the graph
                # laplacian) because the circular graph won't show those anyway
                # (they're not edges).
                unique_edge_values = get_halfvec(lambda_sq, k=1)
                extreme = np.abs(unique_edge_values).max()
                # prevent full connectivity maps from being too messy to
                # interpret
                n_lines = (
                    None if use_edge_rois else
                    int(np.rint(strongest_edges * unique_edge_values.size)))
                if cond_name == 'all_conds':
                    # discard the sign; this is not a difference between
                    # conditions so the sign is not meaningful (all edges will
                    # be negative due to how the graph laplacian is defined)
                    lambda_sq = np.abs(lambda_sq)
                    cmap = 'Blues'
                    vlims = dict(vmin=0, vmax=extreme)
                else:
                    cmap = 'RdBu'
                    vlims = dict(vmin=-extreme, vmax=extreme)
                fig, ax = plot_connectivity_circle(
                    lambda_sq.values, lambda_sq.coords['region_1'].values,
                    n_lines=n_lines, node_angles=node_angles,
                    node_colors=node_colors, title='', interactive=False,
                    show=False, facecolor='w', textcolor='k',
                    node_edgecolor='none', colormap=cmap, colorbar_size=0.4,
                    colorbar_pos=(1., 0.1), linewidth=1, **vlims)
                fig.set_size_inches(8, 8)
                if cond_name == 'all_conds':
                    measure = 'strength of'
                    subtitle = '(all conditions pooled)'
                else:
                    measure = 'contribution to difference in'
                    subtitle = '(attend minus ignore)'
                pctile = ('ROI edges only' if use_edge_rois else
                          'strongest '
                          f'{int(100 * strongest_edges)}% of edges.')
                title = (
                    f'Relative {measure} network connectivity {subtitle},\n'
                    f'{pctile}\n'
                    f'{parcellation} parcellation, {freq_band} band')
                fig.suptitle(title)
                for _ax in fig.axes:
                    if _ax == ax:
                        continue
                    _ax.set_ylabel('Arbitrary units')
                fpath = os.path.join(plot_dir,
                                     f'{slug}-connectivity-circle.pdf')
                fig.savefig(fpath)
                plt.close(fig)
