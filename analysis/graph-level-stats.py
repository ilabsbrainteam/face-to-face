#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute graph-level stats.

authors: Daniel McCloy
license: MIT
"""

from functools import reduce
import os
import yaml

import numpy as np
from scipy.stats import chi2
from scipy.special import comb
from scipy.linalg import sqrtm
import xarray as xr
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from f2f_helpers import load_paths, load_subjects, load_params, get_halfvec

# flags
n_sec = 7  # epoch duration to use
print_results = False

# enable interface to R
corpcor = importr('corpcor')
numpy2ri.activate()

# config paths
data_root, subjects_dir, results_dir = load_paths()
*_, results_root_dir = load_paths(include_inv_params=False)
epo_dir = os.path.join(results_root_dir, 'epochs')
param_dir = os.path.join('..', 'params')
xarray_dir = os.path.join(results_dir, 'xarrays')
stats_dir = os.path.join(results_dir, 'stats', 'graph-level')
plot_dir = os.path.join(results_dir, 'figs', 'graph-metrics')
for _dir in (stats_dir, plot_dir,):
    os.makedirs(_dir, exist_ok=True)

# load other config values
subjects = load_subjects()
surrogate = load_params(os.path.join(param_dir, 'surrogate.yaml'))
excludes = load_params(os.path.join(epo_dir, 'not-enough-good-epochs.yaml'))
roi_edge_dict = load_params(os.path.join(param_dir, 'roi-edges.yaml'))
analysis_bands = load_params(os.path.join(param_dir, 'analysis_bands.yaml'))

# how many subjs are available at this epoch length?
this_excludes = {subj for subj in excludes if n_sec in excludes[subj]}
this_subjects = sorted(set(subjects) - this_excludes)
n_subj = len(this_subjects)

# loop over models
for parcellation, roi_edges in roi_edge_dict.items():
    roi_nodes = sorted(set(reduce(tuple.__add__, roi_edges)))
    # loop over analysis bands
    for freq_band in analysis_bands:
        # load xarray
        fname = f'{parcellation}-{n_sec}sec-{freq_band}-band-graph-metrics.nc'
        graph_metrics = xr.load_dataarray(os.path.join(xarray_dir, fname))
        # separate the "allconds" condition
        allconds = graph_metrics.loc[['allconds']]
        graph_metrics = graph_metrics.loc[['attend', 'ignore']]
        # loop over xarrays: all trials, or attend-vs-ignore comparison
        for cond_name, _xarray in dict(all_conds=allconds,
                                       attend_vs_ignore=graph_metrics).items():
            # loop over choice to use all nodes vs only ROI nodes
            for use_edge_rois in (False, True):
                n_nodes = (len(roi_nodes) if use_edge_rois else
                           _xarray.coords['region_1'].size)
                # compute mean laplacians
                mean_over_subj = _xarray.mean(dim='subject')
                mean_laplacians = mean_over_subj.loc[:, 'graph_laplacian']
                # absence of off-diagonal structural zeros is an assumption of
                # the method
                for _ml in mean_laplacians:
                    offdiag = _ml.values[np.tril_indices_from(_ml, k=-1)]
                    assert np.all(offdiag != 0)
                # if pooling all trials (allconds), turn np.diff() into a no-op
                # by passing n=0 (otherwise it returns an empty array)
                n_diff = 0 if cond_name == 'all_conds' else 1
                # reduce matrices to half-vectorized form (upper triangle
                # including main diagonal; AKA remove redundant entries)
                mean_halfvecs = np.array(
                    [get_halfvec(_ml) for _ml in mean_laplacians])
                mean_diff_halfvec = np.diff(mean_halfvecs, n=n_diff, axis=0
                                            ).squeeze()
                subj_laplacians = _xarray.loc[:, :, 'graph_laplacian']
                subj_diffs = np.diff(subj_laplacians, n=n_diff, axis=0
                                     ).squeeze()
                subj_diff_halfvecs = np.array(
                    [get_halfvec(arr) for arr in subj_diffs])
                # restrict halfvecs to just the ROI edges (if desired)
                if use_edge_rois:
                    halfvec_mask = get_halfvec(
                        mean_over_subj[0].loc['orthog_proj_mat']
                        ).astype(bool)
                    assert halfvec_mask.sum() == len(roi_edges)
                else:
                    halfvec_mask = slice(None)
                subj_diff_halfvecs = subj_diff_halfvecs[:, halfvec_mask]
                mean_diff_halfvec = mean_diff_halfvec[halfvec_mask]
                # call out to R to use special shrinkage methods designed for
                # graphs; use corpcor.cov_shrink(...) instead if we needed
                # sigma (not just its inverse)
                sigma_inv = corpcor.invcov_shrink(subj_diff_halfvecs)
                # TODO for general implementation: include Higham 2002
                # algorithm that ensures positive definiteness / uses nearest
                # (by Frobenius norm) positive definite neighbor. Pseudocode:
                #     ∆S_0 = 0, Y_0 = A
                #     for k = 1, 2, . . .
                #         R_k = Y_{k−1} − ∆S_{k−1}
                #         X_k = P_S (R_k)
                #         ∆S_k = X_k − R_k
                #         Y_k = P_U (X_k)
                #     end

                # calculate test statistic
                statistic = n_subj * (mean_diff_halfvec @ sigma_inv @
                                      mean_diff_halfvec)
                degrees_of_freedom = comb(n_nodes, 2, exact=True)
                pval = 1 - chi2.cdf(statistic, df=degrees_of_freedom)
                # estimate contribution of each edge to the difference betw.
                # conditions (lambda_sq_vec)
                sqrt_inv_sigma = sqrtm(sigma_inv)
                lambda_hat = sqrt_inv_sigma @ mean_diff_halfvec
                lambda_sq_vec = lambda_hat ** 2
                # sanity check: alternate computation of test statistic
                # (Ginestet et al 2017, section 4.4)
                decimal = 5 if cond_name == 'all_conds' else 7
                np.testing.assert_almost_equal(
                    statistic, n_subj * np.sum(lambda_sq_vec), decimal=decimal)
                # populate `signed_lambda_sq` matrix (when loaded in it's just
                # zeros, but conveniently it has all the right axis labels).
                # Note: we're preserving the sign of lambda_hat to keep track
                # of directionality of effect for each edge, for the
                # attend-minus-ignore analysis.
                signed_lambda_sq = (mean_over_subj.loc[:, 'lambda_squared']
                                                  .mean(dim='condition'))
                triu_indices = np.triu_indices_from(signed_lambda_sq)
                if use_edge_rois:
                    triu_indices = tuple(ixs[halfvec_mask]
                                         for ixs in triu_indices)
                signed_lambda_sq_vec = np.sign(lambda_hat) * lambda_sq_vec
                signed_lambda_sq.values[triu_indices] = signed_lambda_sq_vec
                signed_lambda_sq.values.T[triu_indices] = signed_lambda_sq_vec
                # make sure we got the right indices
                if use_edge_rois:
                    for dim, region in enumerate(signed_lambda_sq.coords.dims):
                        indices = np.nonzero(signed_lambda_sq)[dim].values
                        nodes = signed_lambda_sq.coords[region].values[indices]
                        assert set(nodes) == set(roi_nodes)
                # sort edges by strength of contribution (increasing)
                ranked_indices = np.argsort(lambda_sq_vec)
                ranked_lambda_hat = lambda_hat[ranked_indices]
                ranked_edge_indices = tuple(
                    x[ranked_indices] for x in triu_indices)
                ranked_edges = list(zip(
                    *(signed_lambda_sq[ranked_edge_indices].coords[reg].values
                      for reg in signed_lambda_sq.coords.dims)))
                # set cutoff for which ones to print
                q = 75 if use_edge_rois else 99
                abs_thresh = np.percentile(lambda_sq_vec, q)
                thresh_ix = np.searchsorted(ranked_lambda_hat ** 2, abs_thresh)
                # print which edges change the most
                if print_results:
                    print(f'TOP {100 - q}% EDGES WITH STRONGEST DIFFERENCE '
                          'BETWEEN CONDITIONS (ATTEND MINUS IGNORE):')
                    # convert thresh_ix from ascending → descending
                    _slice = slice(None, thresh_ix, -1)
                    for _nodes, _hat in zip(ranked_edges[_slice],
                                            ranked_lambda_hat[_slice]):
                        contrib = np.sign(_hat) * (_hat ** 2)
                        print(f'{_nodes[0]:27} ←→ {_nodes[1]:>27}'
                              f'    {round(contrib, 3):<+06.03}')

                # save stats info to file
                edge_contributions = {
                    f'{" <--> ".join(_edge)}': float(_hat)
                    for _edge, _hat in zip(
                        ranked_edges[::-1], ranked_lambda_hat[::-1])
                }
                output = dict(
                    statistic=float(statistic),
                    degrees_of_freedom=degrees_of_freedom,
                    p_value=float(pval),
                    lambda_hat=edge_contributions
                )
                roi = 'roi' if use_edge_rois else 'all'
                slug = (f'{cond_name}-{parcellation}-{n_sec}sec-'
                        f'{freq_band}-band-{roi}-edges')
                fname = f'{slug}-graph-level-stats.yaml'
                with open(os.path.join(stats_dir, fname), 'w') as f:
                    yaml.dump(output, f)
                # save lambda hat vector and lambda squared matrix
                coords = {f'region_{ix + 1}': ('edge', list(region))
                          for ix, region in enumerate(zip(*ranked_edges))}
                lambda_hat_xarray = xr.DataArray(ranked_lambda_hat,
                                                 coords=coords,
                                                 dims=['edge'],
                                                 name='lambda hat')
                fname = f'{slug}-lambda-hat.nc'
                lambda_hat_xarray.to_netcdf(os.path.join(xarray_dir, fname))
                fname = f'{slug}-lambda-sq.nc'
                signed_lambda_sq.to_netcdf(os.path.join(xarray_dir, fname))
