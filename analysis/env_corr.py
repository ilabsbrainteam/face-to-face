#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Envelope correlation of face-to-face data (for connectivity analysis). The data
are expected to be already band-pass filtered (theta band) and epoched.

authors: Daniel McCloy
license: MIT
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne_connectivity import envelope_correlation
from mne_connectivity.viz import plot_connectivity_circle
import mnefun
from f2f_helpers import load_paths, load_subjects, load_params, get_roi_labels

# flags
force_recompute = False  # set False to quickly iterate through plotting tweaks
plot_circular = True

# config paths
data_root, subjects_dir, results_dir = load_paths()
param_dir = os.path.join('..', 'params')
corr_dir = os.path.join(results_dir, 'envelope-correlations')
for _dir in (corr_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
n_epochs = load_params(os.path.join('..', 'params', 'min_epochs.yaml'))
inv_params = load_params(os.path.join(param_dir, 'inverse_params.yaml'))
orientation_constraint = (
    '' if inv_params['orientation_constraint'] == 'loose' else
    f"-{inv_params['orientation_constraint']}")

mnefun_params_fname = os.path.join('..', 'preprocessing', 'mnefun_params.yaml')
mnefun_params = mnefun.read_params(mnefun_params_fname)
lp_cut = int(mnefun_params.lp_cut)

# initialize figure
fig = plt.figure(figsize=(25, 20), constrained_layout=True)
subplot_shape = (4, 5)
axs = fig.subplots(*subplot_shape, sharex=True, sharey=True)

# handle colormap lims
cmap = get_cmap('viridis').copy()
cmap.set_extremes(under='white', over='white')
cmap_minmax_fname = 'correlation-min-max.yaml'
cmap_minmax_exist = os.path.exists(cmap_minmax_fname)
if cmap_minmax_exist:
    clim = load_params(cmap_minmax_fname)
else:
    print('WARNING: colormap limits will default to 0, 1')
    clim = (0, 1)

for ix, (subj, ax) in enumerate(zip(subjects, axs.ravel()), start=1):
    # see if we've already computed it
    corr_fname = f'{subj}-theta-band-envelope-correlation.npy'
    corr_fpath = os.path.join(corr_dir, corr_fname)
    corr_exist = os.path.exists(corr_fpath)

    # load labels
    labels = get_roi_labels(subj, param_dir)
    label_names = [lab.name for lab in labels]
    if force_recompute or not corr_exist:
        # load epochs
        _dir = os.path.join(data_root, subj)
        fname = f'All_{lp_cut}-sss_{subj}-epo.fif'
        epochs = mne.read_epochs(os.path.join(_dir, 'epochs', fname),
                                 preload=False)
        # truncate to same number of epochs per subject
        new_eve = np.empty((0, 3), dtype=int)
        event_codes = list(epochs.event_id.values())
        for code in event_codes:
            mask = epochs.events[:, -1] == code
            new_eve = np.vstack((new_eve, epochs.events[mask][:n_epochs]))
        new_eve = new_eve[np.argsort(new_eve[:, 0])]
        selection = np.nonzero(np.in1d(epochs.events[:, 0], new_eve[:, 0]))[0]
        epochs = epochs[selection]
        # load inverse
        inv_fname = f'{subj}-{lp_cut}-sss-meg{orientation_constraint}-inv.fif'
        inv_fpath = os.path.join(_dir, 'inverse', inv_fname)
        inv_operator = read_inverse_operator(inv_fpath)
        # apply inverse
        snr = 1.0  # assume lower SNR for single epochs
        lambda2 = 1.0 / snr ** 2
        method = inv_params['method']
        pick_ori = (None if inv_params['estimate_type'] == 'magnitude' else
                    inv_params['estimate_type'])
        stcs = apply_inverse_epochs(epochs, inv_operator, lambda2, method,
                                    pick_ori=pick_ori, return_generator=True)
        # get avg signal in each label (mean_flip reduces signal cancellation)
        src = inv_operator['src']
        label_timeseries = mne.extract_label_time_course(
            stcs, labels, src, mode='mean_flip', return_generator=True)
        # compute envelope correlation
        epo_corr = envelope_correlation(label_timeseries, names=label_names)
        corr_mat = np.squeeze(epo_corr.combine('mean').get_data())
        # save results
        fname = f'{subj}-theta-band-envelope-correlation.npy'
        np.save(os.path.join(corr_dir, fname), corr_mat)
    else:
        corr_mat = np.load(corr_fpath)
    # determine optimal colormap lims
    clim[1] = max(clim[1], corr_mat.max())
    clim[0] = min(clim[0], corr_mat[np.tril_indices_from(corr_mat, -1)].min())
    if plot_circular:
        # prepare label names
        label_colors = [lab.color for lab in labels]
        lh_names = [lab for lab in label_names if lab.endswith('lh')]
        rh_names = [lab for lab in label_names if lab.endswith('rh')]
        label_order = lh_names[::-1] + rh_names
        node_angles = mne.viz.circular_layout(
            label_names, label_order, start_pos=90,
            group_boundaries=[0, len(label_names) / 2])
        # show only strongest ~10% of labels
        n_lines = np.rint(len(labels) * (len(labels) - 1) / 2 / 10).astype(int)
        # plot
        plot_connectivity_circle(
            corr_mat, label_names, n_lines=n_lines, node_angles=node_angles,
            node_colors=label_colors, title='', fig=fig,
            subplot=(*subplot_shape, ix), facecolor='w', textcolor='k',
            node_edgecolor='w', colorbar=False, colormap='viridis',
            vmin=clim[0], vmax=clim[1], show=False)
    else:
        ax.imshow(corr_mat, cmap=cmap, clim=clim)
# add global colorbar
if plot_circular:
    fig.suptitle('Envelope correlation')
else:
    fig.colorbar(ax.images[0], ax=axs, shrink=0.5)


# store computed colormap lims for next time
with open(cmap_minmax_fname, 'w') as f:
    yaml.dump([float(clim[0]), float(clim[1])], f)

# clean up extra axes
for ax in axs.ravel():
    if not len(ax.images):
        ax.set_axis_off()

fig_fname = f"connectivity-{'circles' if plot_circular else 'matrices'}.png"
fig.savefig(fig_fname)
