#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:50:46 2023

@author: jamescallaghan
"""

import os
import wrds
import datetime
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *
from tqdm import tqdm
import datetime as dt

import pandas as pd
from pandas.tseries.offsets import MonthEnd
from pandas.tseries.offsets import MonthBegin

os.chdir('/Users/jamescallaghan/Desktop/IMPERIALTHESIS2')
data_path = os.path.join('30_DEC')

crsp_s = pd.read_csv(os.path.join(data_path,'crsp_esg_30.csv')).drop(labels = ['Unnamed: 0'], axis = 'columns')

# %%

crsp_m = crsp_s.copy()

#set to ESG (ESG Score), ES (Environmental Score), SS (Social Score), GS (Governance Score)

subset = 'GS_DUMMY'


#set to high (1) or low (0) scores based on median value per date
high_or_low = 0

# %%

# create a Boolean mask for the rows to keep
mask = crsp_m[subset] == high_or_low

# select all rows except the ones that do not contain 0
crsp_s = crsp_m[mask].copy()

# %%

crsp_s.to_csv(f'17_crsp_{subset}_{high_or_low}.csv') #save the data 
