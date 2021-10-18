#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute connectivity degree and plot it.

authors: Daniel McCloy
license: MIT
"""

import os
import yaml
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.image import imread
import matplotlib.pyplot as plt
import mne
import mne_connectivity
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params


def get_slug(subj, band, cond):
    return f'{subj}-{condition}-{freq_band}-band'


# flags
freq_bands = ('delta', 'theta', 'beta')
conditions = ('attend', 'ignore', 'attend-ignore')
parcellation = 'aparc'
threshold_prop = 0.15
force_rerender = True  # set False for tweaking overview plotting

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
conn_dir = os.path.join(results_dir, 'envelope-correlations')
plot_dir = os.path.join(results_dir, 'figs', 'degree')
for _dir in (plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
inv_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
orientation_constraint = (
    '' if inv_params['orientation_constraint'] == 'loose' else
    f"-{inv_params['orientation_constraint']}")
cov_type = inv_params['cov_type']  # 'erm' or 'baseline'

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)

# handle colormap lims
cmaps = dict(positive=get_cmap('viridis').copy(),
             symmetric=get_cmap('RdBu_r').copy())
clims = {key: np.array([0, 0]) for key in cmaps}
for key, cmap in cmaps.items():
    cmap.set_extremes(under='black', over='black')
    cmap_minmax_fname = f'degree-{key}-colormap-min-max.yaml'
    if os.path.exists(cmap_minmax_fname):
        clims[key] = np.array(load_params(cmap_minmax_fname))
    else:
        print('WARNING: colormap limits will not be consistent across plots')

# load labels
roi_names = load_params(os.path.join(param_dir, 'rois.yaml'))
labels_to_skip = load_params(
    os.path.join(param_dir, 'skip_labels.yaml'))[parcellation]
roi_names = sorted(set(roi_names) - set(labels_to_skip))
regexp = '|'.join(roi_names)
labels = mne.read_labels_from_annot(surrogate, parc=parcellation,
                                    regexp=regexp, subjects_dir=None)
label_names = [label.name for label in labels]

if force_rerender:
    for subj in subjects:
        # load source space (from inverse)
        inv_fnames = dict(
            erm=f'{subj}-meg-erm{orientation_constraint}-inv.fif',
            baseline=f'{subj}-{lp_cut}-sss-meg{orientation_constraint}-inv.fif'
        )
        inv_fname = inv_fnames[cov_type]
        inv_fpath = os.path.join(data_root, subj, 'inverse', inv_fname)
        source_space = mne.minimum_norm.read_inverse_operator(inv_fpath)['src']

        # load connectivity
        for freq_band in freq_bands:
            degrees = dict()
            for condition in conditions:
                if condition in ('attend', 'ignore'):
                    slug = get_slug(subj, freq_band, condition)
                    conn_fname = (
                        f'{parcellation}-{slug}-envelope-correlation.nc')
                    conn_fpath = os.path.join(conn_dir, conn_fname)
                    conn = mne_connectivity.read_connectivity(conn_fpath)
                    full_degree = mne_connectivity.degree(conn, threshold_prop)
                    # restrict to ROIs of interest
                    roi_indices = np.in1d(conn.names, label_names)
                    degrees[condition] = full_degree[roi_indices]
                    # determine optimal colormap lims
                    clims['positive'] = np.array([
                        min(clims['positive'][0], degrees[condition].min()),
                        max(clims['positive'][1], degrees[condition].max())
                    ])
                else:
                    contrast = condition.split('-')
                    slug = get_slug(subj, freq_band, '-minus-'.join(contrast))
                    degree = np.subtract(*(degrees[cond] for cond in contrast))
                    # determine optimal colormap lims
                    clims['symmetric'] = np.array([
                        min(clims['symmetric'][0], np.abs(degree).min()),
                        max(clims['symmetric'][1], np.abs(degree).max())
                    ])
                    degrees[condition] = degree
                # convert to STC
                # XXX stc = mne.labels_to_stc(roi_labels, degrees[condition])
                stc = mne.labels_to_stc(labels, degrees[condition])
                stc = stc.in_label(
                    mne.Label(source_space[0]['vertno'], hemi='lh') +
                    mne.Label(source_space[1]['vertno'], hemi='rh'))
                # handle colormap limits
                brain_clim = dict(kind='value')
                if condition in ('attend', 'ignore'):
                    plot_key, clim_key = 'lims', 'positive'
                else:
                    plot_key, clim_key = 'pos_lims', 'symmetric'
                brain_clim.update({plot_key: (clims[clim_key][0],
                                              clims[clim_key].mean(),
                                              clims[clim_key][1])})
                # plot
                brain = stc.plot(
                    # clim=dict(kind='percent', lims=[65, 80, 95]),
                    clim=brain_clim, colormap=cmaps[clim_key],
                    subjects_dir=subjects_dir,
                    views=['lateral', 'medial'], hemi='split',
                    smoothing_steps='nearest', time_label=f'{freq_band} band',
                    size=(900, 700), background='w', title=slug)
                # save
                img_fname = f'{slug}-degree.png'
                brain.save_image(os.path.join(plot_dir, img_fname))
                brain.close()
                del brain

# make composite figures
for freq_band in freq_bands:
    for condition in conditions:
        fig = plt.figure(figsize=(25, 20), constrained_layout=True)
        subplot_shape = (4, 5)
        axs = fig.subplots(*subplot_shape, sharex=True, sharey=True)
        # loop over subjects
        for ix, (subj, ax) in enumerate(zip(subjects, axs.ravel()), start=1):
            # load and plot image
            slug = get_slug(subj, freq_band, condition)
            img_fname = f'{slug}-degree.png'
            img = imread(os.path.join(plot_dir, img_fname))
            ax.imshow(img)
            ax.set_title(subj)
        # clean up extra axes
        for ax in axs.ravel():
            if not len(ax.images):
                ax.set_axis_off()
        # add title
        cond_title = (condition if condition in ('attend', 'ignore') else
                      '-minus-'.join(cond for cond in condition.split('-')))
        title = (f'Connectivity degree in the {cond_title} condition '
                 f'({freq_band} band)')
        fig.suptitle(title)
        # save figure
        fig_fname = f'degree-{freq_band}-band-{condition}.png'
        fig_fpath = os.path.join(plot_dir, '..', fig_fname)
        fig.savefig(fig_fpath)

# store computed colormap lims for next time
for key, clim in clims.items():
    cmap_minmax_fname = f'degree-{key}-colormap-min-max.yaml'
    with open(cmap_minmax_fname, 'w') as f:
        yaml.dump([float(clim[0]), float(clim[1])], f)
