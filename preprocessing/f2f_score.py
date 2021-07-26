#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Score function (experimental event categorizer) for mnefun preprocessing of
face-to-face data (for connectivity analysis). This particular score function
does three things:

- ignores event code "1" (representing XXX)
- converts each event code "3" or "5" into a sequence of equally-spaced events
  (original "trials" were 7 seconds long, we're converting each to a series of
  overlapping epochs)
- handles subject f2f_108 differently, as the stim channel triggers were
  configured wrongly for that subject

authors: Daniel McCloy
license: MIT
"""

import numpy as np
import mne
from mnefun._paths import (get_raw_fnames, get_event_fnames)

# INCOMING TRIGGERS
# =================
# 1 (STI001) = "ready" light?
# 2 (STI002) = trial start ("attend" condition)
# 4 (STI003) = trial start ("ignore" condition)

# INCOMING EVENT CODES
# ====================
# 1 = "ready" light? (do not use)
# 3 = trial start ("attend" condition)
# 5 = trial start ("ignore" condition)

# config
orig_dur = 7   # original trial duration
new_dur = 1    # desired epoch duration
spacing = 0.5  # epoch spacing (in seconds)
n_epochs = int((orig_dur - new_dur) // spacing) + 1
offsets = np.linspace(0, orig_dur - new_dur, n_epochs)


def f2f_score(p, subjects):
    for si, subject in enumerate(subjects):
        fnames = get_raw_fnames(p, subject, which='raw', erm=False,
                                add_splits=False, run_indices=None)
        event_fnames = get_event_fnames(p, subject, run_indices=None)
        for fname, event_fname in zip(fnames, event_fnames):
            raw = mne.io.read_raw_fif(fname, allow_maxshield=True)
            sfreq = raw.info['sfreq']
            events = mne.find_events(raw, shortest_event=1)
            # discard 1-triggers. note that one subject had different
            # triggering (8s instead of 2s, not sure why)
            valid_events = (9, 5) if subject == 'f2f_108' else (3, 5)
            mask = np.in1d(events[:, 2], valid_events)
            new_events = np.empty((0, 3))
            for row in events[mask]:
                new_event_code = 55 if row[-1] == 5 else 31
                offset_samp = np.rint(offsets * sfreq).astype(int)
                offset_samp = np.stack([offset_samp], axis=1)
                offset_events = np.hstack((
                    offset_samp + row[0],
                    np.zeros_like(offset_samp),
                    np.full_like(offset_samp, new_event_code)
                ))
                new_events = np.vstack((new_events, offset_events))
            # make sure subsequent trials don't overlap too much
            assert np.all(np.diff(new_events[:, 0]) >=
                          np.floor(spacing * sfreq))
            mne.write_events(event_fname, new_events)
