#!/bin/bash
python compute-connectivity.py      && \
python plot-thresholds-grandavg.py  && \
python plot-label-correlation.py    && \
python make-xarrays.py              && \
python graph-level-stats.py         && \
python plot-circular-graph.py       && \
python plot-graph-level-stats.py    && \
python node-level-stats.py          && \
python plot-node-level-stats.py     && \
python correlate-behavior.py
