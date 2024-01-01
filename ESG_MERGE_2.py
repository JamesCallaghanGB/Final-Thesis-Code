#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:50:47 2023

@author: jamescallaghan
"""

# some of the code adapted from Qingyi (Freda) Song Drechsler (2020)
# https://wrds-www.wharton.upenn.edu/pages/grid-items/momentum-strategy/
# https://wrds-www.wharton.upenn.edu/documents/1442/wrds_momentum_demo.html

# library imports
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

# set current working dirrectory
os.chdir('/Users/jamescallaghan/Desktop/IMPERIALTHESIS2/')
# %%
# load wrds data: set to true, only reuse SQL query if want to test unclean data
load_data_from_csv = True
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
                      
                      where a.date between '08/01/2009' and '12/31/2022'
                      and b.exchcd between -2 and 2
                      and b.shrcd between 10 and 11
                      """, date_cols=['date'])
    
     #crsp_m.to_csv('crsp_m.csv') #save the data 

elif load_data_from_csv:
    # load securities & ESG data
    data_path = os.path.join('27_DEC')
    
    #crsp_m read from disc after running through data cleaning
    crsp_m = pd.read_csv(os.path.join(data_path,'crsp_final_27.csv'))
    
    #read esg data from disc, no neat SQL querry
    esg = pd.read_excel(os.path.join(data_path, 'ESG_DB_USA_16.xlsx')) 
    esg.rename(columns = {'Date':'date','TICKER':'ticker','Total ESG Score':'ESG', 'Governance Score':'GS', 'Environment Score': 'ES', 'Social Score': 'SS'}, inplace = True)

    #size, frequency, stats of dataframes
    print(f'There are {len(crsp_m)} and {len(esg)} rows in the crsp_m and esg datasets, respectively.')
    print('The number of unique tickers by day is: ',crsp_m.groupby(['date'])['ticker'].nunique(),'\n')
    print(stats.describe(crsp_m.groupby(['date'])['ticker'].nunique())) #min,max,mean etc of the number of unique daily tickers

first_check = esg.groupby(['date'])['ticker'].nunique()

esg = esg.sort_values(['ticker', 'date'], ignore_index=True)

#shift scores by 6 months to allow market to 
esg['ESG'] = esg.groupby('ticker')['ESG'].shift(6)
esg['ES'] = esg.groupby('ticker')['ES'].shift(6)
esg['SS'] = esg.groupby('ticker')['SS'].shift(6)
esg['GS'] = esg.groupby('ticker')['GS'].shift(6)

#there is a jump in 2016 in the number of tickers, this jump is present in when 
#sampling the entire database, so is not unique to this study, likely as a 
#result of the purchase of sustainalytics by morningstar, and their prior 
#relationship between databases

count_esg = esg.groupby('date')['ESG'].nunique()
count_esg.plot()

print('Number of crsp_m rows with no ticker data', crsp_m.ticker.isna().sum())
#crsp_m = crsp_m[crsp_m['ticker'].notna()] #drop entries with missing tickers

#drop 'Unnamed: 0' column remnant from crsp_m if necessary
#crsp_m = crsp_m.drop(labels = ['Unnamed: 0'], axis = 'columns')

# %%

crsp_m_tickers = set(crsp_m.ticker.unique())
print('Number of tickers in crsp_m', len(crsp_m_tickers))

esg_ticker_index = set(esg['ticker'].unique())
print('Number of tickers in esg', len(esg_ticker_index))

intersection = esg_ticker_index.intersection(crsp_m_tickers)
missing_from_esg = list(esg_ticker_index ^ set(intersection))
missing_from_crsp = list(crsp_m_tickers ^ set(intersection))

intersection = list(intersection)


# %%

#check esg for missing data
print('Number of rows in esg with missing ticker data', esg.ticker.isna().sum())

#change crsp_m date to datetime, esg date is already in datetime format, but if needed can manually set
crsp_m.date = pd.to_datetime(crsp_m.date) 

#for some reason this is the only way to change recorded dates to end of month
crsp_m = crsp_m.set_index(['date'])
crsp_m.index = crsp_m.index.to_period('M').to_timestamp('M')
crsp_m = crsp_m.reset_index()

#for some reason this is the only way to change recorded dates to end of month
esg = esg.set_index(['date'])
esg.index = esg.index.to_period('M').to_timestamp('M')
esg = esg.reset_index()

print('esg columns:', list(esg.columns.values))
esg = esg[['date','ticker','Company', 'ESG', 'ES', 'SS', 'GS']]

#take only those firms in the intersection of the two databases
esg = esg[esg['ticker'].isin(intersection)]
crsp_m = crsp_m[crsp_m['ticker'].isin(intersection)]

# %%

#create dictionary to join on crsp
to_duplicate_esg = esg.set_index(['date','ticker'])
crsp_y = crsp_m.set_index(['date','ticker']).copy()

#join on the calling frame's index
crsp_y = crsp_y.join([to_duplicate_esg], how='left').reset_index()
crsp_y = crsp_y.sort_values(['permno','date'])

#check no values in columns pre start date of the esg database (08-2009)
ESG_check = crsp_y.groupby('date')['ESG'].nunique()

print('Number of tickers crsp_y', crsp_y.ticker.nunique())
print('Number of permnos crsp_y', crsp_y.permno.nunique())

#count = crsp_y.groupby('ticker')['permno'].nunique()
#ticker_count = crsp_y.groupby(['date', 'ticker'])['ticker'].nunique()

#can merge, but removes all securities data missing an esg score, and does not
#allocate score to correct date, or allow esg score to be rolled forward as  
#done in the next chunk

# %%

#this creates a new frame for each company, rolls date forward, then appends all
#individual company frames together to end in the original panel data fromat

l = []

for group, subdf in crsp_y.sort_values(['ticker', 'date']).groupby('ticker'):
    values_for_subdf_ESG = subdf[['ESG']].ffill(axis = 0, limit = 12)
    values_for_subdf_ES = subdf[['ES']].ffill(axis = 0, limit = 12)
    values_for_subdf_SS = subdf[['SS']].ffill(axis = 0, limit = 12)
    values_for_subdf_GS = subdf[['GS']].ffill(axis = 0, limit = 12)
    new_subdf = subdf.copy()
    new_subdf.ESG = values_for_subdf_ESG
    new_subdf.ES = values_for_subdf_ES
    new_subdf.SS = values_for_subdf_SS
    new_subdf.GS = values_for_subdf_GS
    l.append(new_subdf)

final = pd.concat(l,axis=0)

final_check = final.groupby('date')['ESG'].nunique()

# %%

# clean up date rangesin order apply qcut function, dates after 09/2020 are cut
# off as there is not enough data after this point to apply qcut
date_range = (final['date'] > '2010-01-31') & (final['date'] <= '2020-09-30')

final = final.loc[date_range]

final_check_2 = final.groupby('date')['ESG'].nunique()

# 0 = low score 1 = high score
final['ESG_DUMMY'] = final.groupby('date')['ESG'].transform(lambda x: pd.qcut(x, 2, labels=False))
final['ES_DUMMY'] = final.groupby('date')['ES'].transform(lambda x: pd.qcut(x, 2, labels=False))
final['SS_DUMMY'] = final.groupby('date')['SS'].transform(lambda x: pd.qcut(x, 2, labels=False))
final['GS_DUMMY'] = final.groupby('date')['GS'].transform(lambda x: pd.qcut(x, 2, labels=False))

# manual check that there is sufficient data per company per month, if n/a -> no
# score is reported that month across any pillar
final['ESG_CHECK'] = final['ESG_DUMMY'] + final['ES_DUMMY'] + final['SS_DUMMY'] + final['GS_DUMMY']
final2 = final[final['ESG_CHECK'].notna()]


# %%

#save data to be used in 'ESG_SUBSETTING.py'
final2.to_csv('crsp_esg_27.csv') #save the data 

