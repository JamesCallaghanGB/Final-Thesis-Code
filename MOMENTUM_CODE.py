#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:43:57 2023

@author: jamescallaghan
"""

# some of the code adapted from Qingyi (Freda) Song Drechsler (2020)
# https://wrds-www.wharton.upenn.edu/pages/grid-items/momentum-strategy/
# https://wrds-www.wharton.upenn.edu/documents/1442/wrds_momentum_demo.html

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

os.chdir('/Users/jamescallaghan/Desktop/IMPERIALTHESIS2/')
data_path = os.path.join('27_DEC')

# %%

load_data_from_csv = True
if not load_data_from_csv:
    conn = wrds.Connection()
    crsp_m = conn.raw_sql("""
                          select a.permno, a.date, a.ret, a.prc, a.shrout, 
                          b.shrcd, b.exchcd, b.comnam
                          
                          from crsp.msf as a
                          left join crsp.msenames as b
                          
                          on a.permno=b.permno
                          and b.namedt<=a.date
                          and a.date<=b.nameendt
                          
                          where a.date between '01/01/2008' and '12/31/2020'
                          and b.exchcd between -2 and 2
                          and b.shrcd between 10 and 11
                          """, date_cols=['date']) 

elif load_data_from_csv:
    high_low = '0'
    esg_es_ss_gs = 'ESG_DUMMY'
    crsp_m = pd.read_csv(os.path.join(data_path,f'17_crsp_{esg_es_ss_gs}_{high_low}.csv')).drop(labels = ['Unnamed: 0'], axis = 'columns')

# %%

#to test whole data - do not run cell above but only the hashed line below
#crsp_m = pd.read_csv(os.path.join(data_path,'crsp_esg_30.csv')).drop(labels = ['Unnamed: 0'], axis = 'columns')

# %%

# Change variable format to int
crsp_m[['permno']] = crsp_m[['permno']].astype(int)

print(crsp_m.groupby(['date']).nunique(['ticker']))

#if testing wrds, hash these out
crsp_m.rename(columns={'prc': 'prccm'}, inplace = True)
crsp_m['prccm'] = crsp_m['prccm'].abs()
crsp_m['mktcap'] = (crsp_m['prccm'] * crsp_m['shrout']).abs()

#rearrange and sort
crsp_m = crsp_m[['permno', 'date', 'ret', 'prccm','bom_mktcap', 'mom_signal']
                ].sort_values(['permno', 'date'])

# fill in missing return with 0
crsp_m['ret'] = crsp_m['ret'].fillna(0)

#create log return column
crsp_m['logret'] = np.log(1+crsp_m['ret'])

# %%

J = 12  # Formation Period Length (This is just for the reader's convenience
# to change the formation period, this must be changed in DATA_CLEANING.py)

number_portfolios = 3 # Number of portfolios

K = 2  # Holding Period Length: K

# %%

###############################################################################
######################## Momentum auxiliary dataframe #########################
###############################################################################

###create temporary crsp dataframe for construction of umd
###_tmp_crsp = crsp_m[['permno','date','ret','logret']].sort_values(['permno','date']).set_index('date')

umd = crsp_m[['permno', 'date', 'mom_signal']].sort_values(['permno','date'])

###calculate the momentum signal skipping one month for reversal
###umd = _tmp_crsp.groupby(['permno'])['logret'].rolling(J-1, min_periods=J-1).sum().reset_index()
###umd = umd.rename(columns={'logret':'sumlogret'})

#create mktcap dictionary to merge to umd
to_duplicate_mktcap = crsp_m[['date','permno','bom_mktcap']].set_index(['date','permno'])

#set umd index
umd = umd.set_index(['date','permno'])

#join market caps and esg portfolios
umd = umd.join([to_duplicate_mktcap], how='inner').reset_index()

#create cumulative returns column
umd['cumret']=np.exp(umd['mom_signal'])-1

#umd = umd.reset_index()
#crsp_m = crsp_m.reset_index()

#drop na on cumret and create portfolio column
umd=umd.dropna(axis=0, subset=['cumret'])

#split dataframe into n groups based on signal
umd['momr'] = umd.groupby('date')['cumret'].transform(lambda x: pd.qcut(x, number_portfolios, labels=False))

#shift portfolios from 0-n to 1-(n+1) (1: losers // (n+1): winners) where n = number of portfolios
umd['momr'] = 1+umd.momr.astype(int)

#rank each company according to its momentum signal
umd['comp_rank'] = umd.groupby(['date'])['cumret'].rank(method='first', ascending = True)

#create average rank dictionary to merge to umd
to_duplicate = umd.groupby(['date'])[['comp_rank']].mean().copy()
to_duplicate.rename(columns={'comp_rank': 'avrnk'}, inplace=True)

#merge
umd = pd.merge(umd, to_duplicate, on = 'date')

#compute cross sectional rank
umd['rmav'] = umd['comp_rank'] - umd['avrnk']

#assign factor long short positions by applying qcut to split portfolio in half 
#by positive weight (1) and negative weight (0)

umd['fctr'] = umd.groupby('date')['rmav'].transform(lambda x: pd.qcut(x, 2, labels=False))

# %%

#set date to datetime
umd['date'] = pd.to_datetime(umd['date'])

#construct holding periods
umd['form_date'] = umd['date']
umd['medate'] = umd['date'] + MonthEnd(0)
umd['hdate1'] = umd['medate'] + MonthBegin(1)
umd['hdate2'] = umd['medate'] + MonthEnd(K)

#rearrange umd
umd = umd[['permno','form_date','momr','fctr','bom_mktcap','rmav','hdate1','hdate2']]

#value weight
umd['w'] = (umd.groupby(['form_date', 'momr'])['bom_mktcap'].transform(lambda x: x/x.sum(min_count=1)))

#factor weight
umd['f'] = (umd.groupby(['form_date', 'fctr'], group_keys=False)['rmav'].transform(lambda x: x/(np.sum(x) * np.sign(np.sum(x)))))

#merging and filtering
fport = pd.merge(crsp_m[['permno', 'date', 'ret']], umd, on=['permno'], how='inner')
fport = fport[(fport['hdate1'] <= fport['date']) & (fport['date'] <= fport['hdate2'])]

#rearrange fport and set date to datetime
fport = fport[['permno','form_date','momr','fctr','hdate1','hdate2','date','ret','w','f']]
fport['date'] = pd.to_datetime(fport['date'])

#value weighting
fport['retw'] = fport['ret'] * fport['w']

#factor weighting
fport['retf'] = fport['ret'] * fport['f']

# %%

#get tbill and set index
TBILL = pd.read_csv(os.path.join(data_path,'US_30_DAY_TBILL.csv'))
TBILL = TBILL.set_index('caldt')
TBILL.index = pd.to_datetime(TBILL.index)
TBILL.index = TBILL.index.to_period('M').to_timestamp('M')

# %%

#value weighting
umd_port_v = fport.groupby(['date','momr','form_date'])['retw'].sum(min_count=1)
unstack_umd_port_v = umd_port_v.groupby(level=[0, 1]).mean().unstack().reset_index()

unstack_umd_port_v['date'] = pd.to_datetime(unstack_umd_port_v['date'])
start_yr = unstack_umd_port_v.date.dt.year.min()
unstack_umd_port_v = unstack_umd_port_v.loc[unstack_umd_port_v.date.dt.year >= start_yr]

unstack_umd_port_v = unstack_umd_port_v.set_index('date')

#mom portfolio
unstack_umd_port_v['mom'] = unstack_umd_port_v[number_portfolios] - unstack_umd_port_v[1]

#tbill indexing
start_date_y = unstack_umd_port_v.index[0]
end_date_y   = unstack_umd_port_v.index[-1]

#change tbill index
TBILL_V = TBILL[start_date_y:end_date_y].copy()

#calculate excess return on the portfolios
unstack_umd_port_v['xs_mom'] = unstack_umd_port_v['mom'] - TBILL_V['t30ret']
unstack_umd_port_v['xs_winners'] = unstack_umd_port_v[number_portfolios] - TBILL_V['t30ret']
unstack_umd_port_v['xs_middle'] = unstack_umd_port_v[2] - TBILL_V['t30ret']
unstack_umd_port_v['xs_losers'] = unstack_umd_port_v[1] - TBILL_V['t30ret']

#compile momentum output
value_mean = unstack_umd_port_v[['xs_winners', 'xs_middle', 'xs_losers', 'xs_mom']].mean().to_frame()
value_mean = value_mean.rename(columns={0:'mean'}).reset_index()
value_mean

#significance testing excess returns (ttest_1samp default is two tailed)
t_v_losers = pd.Series(stats.ttest_1samp(unstack_umd_port_v['xs_losers'],0.0)).to_frame().T
t_v_middle = pd.Series(stats.ttest_1samp(unstack_umd_port_v['xs_middle'],0.0)).to_frame().T
t_v_winners = pd.Series(stats.ttest_1samp(unstack_umd_port_v['xs_winners'],0.0)).to_frame().T
t_v_long_short = pd.Series(stats.ttest_1samp(unstack_umd_port_v['xs_mom'],0.0)).to_frame().T

t_v_losers['momr']='xs_losers'
t_v_middle['momr'] = 'xs_middle'
t_v_winners['momr']='xs_winners'
t_v_long_short['momr']='xs_mom'

t_v_output = pd.concat([t_v_winners, t_v_middle, t_v_losers, t_v_long_short])\
    .rename(columns={0:'t-stat', 1:'p-value'})
    
value_output = pd.merge(value_mean, t_v_output, on=['momr'], how='inner')

print(value_output)
print('Value Weighted P1 Standard Deviation:', unstack_umd_port_v[1].std())
print('Value Weighted P2 Standard Deviation:', unstack_umd_port_v[2].std())
print('Value Weighted P3 Standard Deviation:', unstack_umd_port_v[3].std())
print('Value Weighted P3-P1 Standard Deviation:', unstack_umd_port_v.mom.std())

# %%

unstack_umd_port_v.to_excel('_____.xlsx', index=True)

# %%

#factor weighting
umd_port_f = fport.groupby(['date','fctr', 'form_date'])['retf'].sum(min_count=1)
unstack_umd_port_f = umd_port_f.groupby(level=[0, 1]).mean().unstack().reset_index()

unstack_umd_port_f['date'] = pd.to_datetime(unstack_umd_port_f['date'])
start_yr = unstack_umd_port_f.date.dt.year.min()
unstack_umd_port_f = unstack_umd_port_f.loc[unstack_umd_port_f.date.dt.year >= start_yr]

unstack_umd_port_f = unstack_umd_port_f.set_index('date')

#factor portfolio
unstack_umd_port_f['mom'] = unstack_umd_port_f[1.0] + unstack_umd_port_f[0.0]

#tbill indexing
start_date_y = unstack_umd_port_f.index[0]
end_date_y   = unstack_umd_port_f.index[-1]

#change tbill index
TBILL_F = TBILL[start_date_y:end_date_y].copy()

#calculate excess return on the portfolios
unstack_umd_port_f['xs_mom'] = unstack_umd_port_f['mom'] - TBILL_V['t30ret']

#compile momentum output
factor_mean = unstack_umd_port_f[['xs_mom']].mean().to_frame()
factor_mean = factor_mean.rename(columns={0:'mean'}).reset_index()
factor_mean

#significance testing excess returns
t_f_long_short = pd.Series(stats.ttest_1samp(unstack_umd_port_f['xs_mom'],0.0)).to_frame().T

t_f_long_short['fctr']='xs_mom'

t_f_output = t_f_long_short.rename(columns={0:'t-stat', 1:'p-value'})

factor_output = pd.merge(factor_mean, t_f_output, on=['fctr'], how='inner')

print(factor_output)
print('Factor Strategy Standard Deviation:', unstack_umd_port_f.mom.std())

# %%

unstack_umd_port_f.to_excel('______.xlsx', index=True)

# %%

import statsmodels.api as sm

#only unhash if running on whole sample, else keep hashed
#high_low = 'whole_sample'
#esg_es_ss_gs = 'whole_sample'

#model_f = sm.OLS(y.values,sm.add_constant(X.values))  # perform linear regression
#results_f = model_f.fit()
#results_f.summary()

FF3F = pd.read_csv(os.path.join(data_path,'F-F_Research_Data_5_Factors_2x3 2.csv'))#,on_bad_lines='skip')

def run_regression(X_values, columns_to_use, y_values,save):
    """X_values and y_values are data frames indexed by the date"""
    
    X_values = X_values[columns_to_use]
    model_   = sm.OLS(y_values, sm.add_constant(X_values.values))
    results_ = model_.fit()
    
    if save == True:
        #save model summary to disk
        file_name_ = f'{esg_es_ss_gs}_{high_low}'
        for col_name in columns_to_use:
            file_name_ += col_name + '_'
        
        with open(f'{file_name_}.txt','w') as fh:
            fh.write(results_.summary().as_text())
    
    print(results_.summary())
    coeff_names_and_values = list(zip(list(X_values.columns), (results_.params)))
    
    print(f'The coefficients for high_low = {high_low} are \n')
    for name, value in coeff_names_and_values:
    
        print(f'{name}: {round(value,5)}')

#initialising timeframe
start_date_y = unstack_umd_port_v.index[0]
end_date_y   = unstack_umd_port_v.index[-1]

#converting variable date to datetime and enduring eom
FF3F.rename(columns = {'Unnamed: 0': 'date'}, inplace = True)
FF3F.date = pd.to_datetime(FF3F.date, format='%Y%m')
FF3F = FF3F.set_index('date')
FF3F.index = FF3F.index.to_period('M').to_timestamp('M')

X = FF3F[start_date_y:end_date_y]

# %%

mom = unstack_umd_port_v['mom']*100 - X.RF

#CAPM: 'Mkt-RF'
run_regression(X_values = X, columns_to_use = ['Mkt-RF'],\
               y_values = mom,save = True )    

#FF5F: 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'
run_regression(X_values = X, columns_to_use = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],\
               y_values = mom,save = True )

# %%

P1 = unstack_umd_port_v[1]*100 - X.RF

#CAPM: 'Mkt-RF'
run_regression(X_values = X, columns_to_use = ['Mkt-RF'],\
               y_values = P1,save = True )    

#FF5F: 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'
run_regression(X_values = X, columns_to_use = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],\
               y_values = P1,save = True )


# %%

P2 = unstack_umd_port_v[2]*100 - X.RF

#CAPM: 'Mkt-RF'
run_regression(X_values = X, columns_to_use = ['Mkt-RF'],\
               y_values = P2,save = True )    

#FF5F: 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'
run_regression(X_values = X, columns_to_use = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],\
               y_values = P2,save = True )


# %%

P3 = unstack_umd_port_v[3]*100 - X.RF

#CAPM: 'Mkt-RF'
run_regression(X_values = X, columns_to_use = ['Mkt-RF'],\
               y_values = P3,save = True )    

#FF5F: 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'
run_regression(X_values = X, columns_to_use = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],\
               y_values = P3,save = True )

# %%

dam = pd.read_csv(os.path.join(data_path,'CRSP_MKT.csv'))#,on_bad_lines='skip')

dam.rename(columns={'caldt': 'date'}, inplace = True)
dam.date = pd.to_datetime(dam.date, format='%Y-%m-%d')
dam = dam.set_index('date')
dam.index = dam.index.to_period('M').to_timestamp('M')

rfr = FF3F[['RF']].copy()
dam_reg = rfr.join([dam], how='inner')

#cumulative returns and making sure variables are on correct scale
dam_reg['xs_mkt'] = dam_reg['vwretd'] - dam_reg['RF']/100
#FF3F['mkt'] = FF3F['Mkt-RF']/100 + FF3F['RF']/100
dam_reg['mkt_cum'] = (dam_reg['vwretd']+1).rolling(window=24).apply(np.prod, raw=True)-1
dam_reg['xs_mkt'] = dam_reg['xs_mkt']*100

#ex ante bear market indicator that takes a value of 1 if the cumulative ... 
#...market return in the preceding 24 months is negative, and 0 otherwise
dam_reg['bearind'] = dam_reg['mkt_cum'] < 0
dam_reg['bearind'] = dam_reg['bearind'].astype(int)

#bull market indicator  defined as 1 - bear market indicator
dam_reg['bullind'] = 1 - dam_reg['bearind']

#upmarket indicator
dam_reg['upmkt'] = dam_reg['xs_mkt'] > 0
dam_reg['upmkt'] = dam_reg['upmkt'].astype(int)

#construct base variables to add to for each regression
OPREG_1 = dam_reg[['xs_mkt']]
OPREG_2 = dam_reg[['xs_mkt', 'bearind']]
OPREG_3 = dam_reg[['xs_mkt', 'bearind', 'upmkt']]
OPREG_4 = dam_reg[['xs_mkt', 'bullind', 'upmkt']]

#constrained capm variable construction
OPREG_2['beartimesmkt'] = OPREG_2['xs_mkt'] * OPREG_2['bearind']

#bear market regression variable construction
OPREG_3['beartimesmkt'] = OPREG_3['xs_mkt'] * OPREG_3['bearind']
OPREG_3['beartimesmkttimesup'] = OPREG_3['xs_mkt'] * OPREG_3['bearind'] * OPREG_3['upmkt']

#bull market regression variable construction
OPREG_4['bulltimesmkt'] = OPREG_4['xs_mkt'] * OPREG_4['bullind']
OPREG_4['bulltimesmkttimesup'] = OPREG_4['xs_mkt'] * OPREG_4['bullind'] * OPREG_4['upmkt']

#optionality regression 2 (constrained capm)
OPREG_2 = OPREG_2[['bearind','xs_mkt','beartimesmkt']]

#optionality regression 3 (bear market)
OPREG_3 = OPREG_3[['bearind','xs_mkt','beartimesmkt', 'beartimesmkttimesup']]

#optionality regression 4 (bull market)
OPREG_4 = OPREG_4[['bullind','xs_mkt','bulltimesmkt', 'bulltimesmkttimesup']]

#drop the risk free rate as a variable for FF5F regressions as needed
#X = X.drop('RF', axis = 1)

# %%   

#zero-investment factor portfolio is used to test in regressions for optionality
z = unstack_umd_port_f['mom']*100

#optionality regressions
for OPREG_i in [OPREG_1,OPREG_2,OPREG_3,OPREG_4]:
    OPREG_i = OPREG_i[start_date_y:end_date_y]
    run_regression(X_values = OPREG_i, columns_to_use  = list(OPREG_i.columns),\
                   y_values = z, save = True)

# %%   

##########   FACTOR    ##########

FACT = unstack_umd_port_f['mom']*100 - FF3F[start_date_y:end_date_y].RF

#CAPM: 'Mkt-RF'
run_regression(X_values = X, columns_to_use = ['Mkt-RF'],\
               y_values = FACT,save = True )    

#FF5F: 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'
run_regression(X_values = X, columns_to_use = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],\
               y_values = FACT,save = True )
    
# %%

#sharpe ratio
sharpe = ((unstack_umd_port_v[number_portfolios] - FF3F['RF']/100).mean())/unstack_umd_port_v[number_portfolios].std()
print('Long-Only H-L Strategy Sharpe Ratio:', sharpe)

#treynor ratio (have to manually input from the CAPM regression output)
beta = 1
treynor = ((unstack_umd_port_v[number_portfolios]*100 - FF3F['RF']).mean())/beta
print('Long-Only H-L Strategy Treynor Ratio:', treynor)

# %%

test = fport.sort_values(['momr','retw'], ascending = False)
unstack_umd_port_v['cumret_winners']   = (1+unstack_umd_port_v[number_portfolios]).cumprod()-1
unstack_umd_port_v['cumret_losers']    = (1+unstack_umd_port_v[1]).cumprod()-1
unstack_umd_port_v['cumret_long_short']= (1+unstack_umd_port_v.mom).cumprod()-1

plt.figure(figsize=(12,9))
plt.suptitle('Low Governance Score', fontsize=20)
ax1 = plt.subplot(211)
ax1.set_title('Long/Short Momentum Strategy', fontsize=15)
ax1.set_xlim([(start_date_y), (end_date_y)])
ax1.plot(unstack_umd_port_v['cumret_long_short'])
ax2 = plt.subplot(212)
ax2.set_title('Cumulative Momentum Portfolios', fontsize=15)
ax2.plot(unstack_umd_port_v['cumret_winners'], 'b-', unstack_umd_port_v['cumret_losers'], 'r--')
ax2.set_xlim([(start_date_y), (end_date_y)])
ax2.legend(('Winners','Losers'), loc='upper left', shadow=True)
plt.subplots_adjust(top=0.92, hspace=0.2)


