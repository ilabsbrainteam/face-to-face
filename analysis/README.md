# Analysis pipeline

- run `compute-connectivity.py` to compute all-to-all envelope connectivity in
  the various frequency bands for a given cortical parcellation (specified in
  the script)
- run `plot-thresholds-grandavg.py` to plot connectivity degree for each
  label for beta-band-filtered data, at a variety of 
  connectivity-degree-inclusion-thresholds
    - examine the resulting plots to pick a final threshold
- run `plot-label-correlation.py` to generate a correlation matrix of label
  connectivity degree (correlation across subjs)
- run `make-xarrays.py` to generate an across-subject data object that combines
  matrices for envelope correlation, adjacency, the graph laplacian, and a
  binary matrix encoding edge ROIs.
- run `graph-level-stats.py` to test whether the connectivity is different
  between "attend" and "ignore" conditions. Results go in `{results_dir}/stats`
- run `plot-circular-graph.py` and `plot-graph-level-stats.py` to show the
  relative contribution of each edge to the difference in connectivity between
  conditions.
- run `node-level-stats.py` to compute and plot some node-level comparisons.
- run `plot-node-level-stats.py` to show which regions change most between
  conditions.
- run once `../behavioral/clean-behavioral-data.py`, then run
  `correlate-behavior.py` to generate correlations between node metrics and
  behavioral measures.
