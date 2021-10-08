# Analysis pipeline

- run `redo-epoching.py` to bandpass at various frequency bands (specified in
  the script) and epoch after bandpassing (so we avoid filter edge artifacts)
- run `compute-connectivity.py` to compute all-to-all envelope connectivity in
  the various frequency bands for a given cortical parcellation (specified in
  the script)
- run `plot-thresholds-grandavg.py` to plot connectivity degree for each
  label for beta-band-filtered data, at a variety of 
  connectivity-degree-inclusion-thresholds
    - examine the resulting plots to pick a final threshold
- run `plot-label-correlation.py` to generate a correlation matrix of label
  connectivity degree (correlation across subjs)
- run `plot-degree.py` to generate per-subject degree plots for the other
  frequency bands
