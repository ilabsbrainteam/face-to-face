#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare behavioral data for correlation with graph metrics.

authors: Daniel McCloy
license: MIT
"""

import pandas as pd

fname = 'F2F_behavioraldata.xlsx'

data = pd.read_excel(fname)

colname_mapper = dict(
    ParticipantId='subj',
    Gender='gender',
    maternalEDU='maternal_education_years',
    SESscore='ses',
    interaction='primary_caregiver_daily_interaction',
    p2interaction='secondary_caregiver_daily_interaction',
    VOCAB_18m='vocab_18',
    VOCAB_21m='vocab_21',
    VOCAB_24m='vocab_24',
    VOCAB_27m='vocab_27',
    VOCAB_30m='vocab_30',
)

data.rename(columns=colname_mapper, inplace=True)
data['subj'] = data['subj'].str.lower()
data = data.filter(items=colname_mapper.values(), axis='columns')
data.to_csv('f2f-behavioral-data.csv', index=False)
