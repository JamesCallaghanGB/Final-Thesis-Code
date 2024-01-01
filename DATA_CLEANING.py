#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:01:16 2023

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

# set current working directory
os.chdir('/Users/jamescallaghan/Desktop/IMPERIALTHESIS2/30_DEC')

# %%
# load wrds data: 
load_data_from_csv = False
if not load_data_from_csv:
    conn = wrds.Connection()
    crsp_m = conn.raw_sql("""
                      select a.permno, a.date, a.ret, a.prc, a.shrout,
                      b.shrcd, b.exchcd, b.ticker
                      
                      from crsp.msf as a
                      left join crsp.msenames as b
                      
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      
                      where a.date between '01/01/2008' and '12/31/2020'
                      and b.exchcd between -2 and 3
                      and b.shrcd between 10 and 11
                      """, date_cols=['date'])

#minimum date of esg database is 08/2009, so collect data before this date
#in order to calculate momentum signal

#crsp_m.to_csv('crsp_raw_data.csv')

# %%

crsp_c = crsp_m.copy()

#rename columns, set prc to absolute value to account for B/A averages...
#...generated by WRDS where closing price not available (denoted by a negative)
crsp_m.rename(columns={'prc': 'prccm'}, inplace = True)
crsp_m['prccm'] = crsp_m['prccm'].abs()
crsp_m['mktcap'] = (crsp_m['prccm'] * crsp_m['shrout']).abs()

crsp_m.sort_values(['permno', 'date'], ignore_index=True)

# %%

#fill missing return with 0 as in WRDS code
crsp_m['ret'] = crsp_m['ret'].fillna(0)

#create log return column
crsp_m['logret'] = np.log(1+crsp_m['ret'])

crsp_m = crsp_m.sort_values(['permno','date']).set_index('date')

J = 12

#calculate the momentum signal skipping one month for reversal
umd = crsp_m.groupby(['permno'])['logret'].rolling(J-1, min_periods=J-1).sum().reset_index()
umd = umd.rename(columns={'logret':'mom_signal'})
umd = umd[['date','permno','mom_signal']].set_index(['date','permno'])

#set crsp_m index
crsp_m = crsp_m.reset_index()
crsp_m = crsp_m.set_index(['date','permno'])

#join mom signals and crsp_m data
crsp_m = crsp_m.join([umd], how='inner').reset_index()

# %%

#check for at least 12 returns over the length of sample
crsp_12 = crsp_m.groupby('permno')['ret'].nunique().reset_index()
mask_12 = crsp_12[crsp_12['ret'] >= 12]
mask_12 = mask_12['permno'].tolist()
crsp_y = crsp_m[crsp_m['permno'].isin(mask_12)]

#check number of unique returns is greater than 12
check_12 = crsp_y.groupby('permno')['ret'].nunique().reset_index()

crsp_p = crsp_y.copy()

#remove penny stocks on a monthly basis
crsp_p['bom_prccm'] = crsp_p.groupby('permno')['prccm'].shift(1)
crsp_p = crsp_p.query('bom_prccm>1').copy()

crsp_p = crsp_p.sort_values(['permno', 'date'], ignore_index = True)

#rank stocks by beginning of month market capitalisation
crsp_p['bom_mktcap'] = crsp_p.groupby('permno')['mktcap'].shift(1)

crsp_z = crsp_p.copy()
crsp_z = crsp_z.sort_values(['date','bom_mktcap'], ascending = False)
crsp_z['mktcap_rank'] = crsp_z.groupby('date')['bom_mktcap'].rank(ascending = False)

to_duplicate_mktcap = crsp_z.groupby(['date'])[['bom_mktcap']].sum().copy()
to_duplicate_mktcap.rename(columns={'bom_mktcap': 'total_mktcap'}, inplace=True)

crsp_z = pd.merge(crsp_z, to_duplicate_mktcap, on = 'date')

crsp_z['mktcap_cum'] = crsp_z.groupby('date')['bom_mktcap'].cumsum()
crsp_z['cum_mktcap_coverage'] = crsp_z['mktcap_cum']/crsp_z['total_mktcap']

crsp_f = crsp_z[crsp_z['cum_mktcap_coverage'] < 0.9].sort_values(['date','mktcap_rank'])

# %%

#save cleaned data to CSV to be used next in 'ESG_MERGE_2.py'
crsp_p.to_csv('crsp_final_30.csv') #save the data 