# Final-Thesis-Code
James Callaghan MSc IWM Thesis: 'ESG &amp; Momentum'

Thank you for taking the time to read my code. To analyse the effect of ESG and 
Momentum, the .py files in this Github recreate the methodologies of Kaiser and 
Welters (2019) and Asness, Pedersen, and Moskowitz (2013), with additional 
moditifcations. 

Instructions:

1: DATA_CLEANING.py - This code uses an SQL querry to collect securities data 
from WRDS CRSP for all common stock on AMEX, NASDAQ, and NYSE exchanges,
excluding ADRs, REITs, foreign firms, etc. The data is then filtered as
outlined in Asness, Pedersen, and Moskowitz (2013).

2: ESG_MERGE_2.py - CRSP and Sustainalytics Data is then merged in such a way
so as to allow ESG scores to be rolled forward and lagged, being treated the 
same as standard accounting variables such as B/M ratios, being valid and
considered not only in the reported month, but quarterly/yearly.

3: ESG_SUBSETTING.py - Mimicks a query function to create subsets of securities
based on ESG pillar information, which are then saved as a CSV to be loaded
into the final momentum code.

4: MOMENTUM_CODE.py - Constructs a set of Value-Weighted and factor-Weighted
portfolios as in Kaiser and Welters (2019) and Asness, Pedersen, and Moskowitz 
(2013). Must run each subset one by one when collecting results. Regressions
are included, but must also be run one by one, as each new regression will over
-write the previous regression.

I am unsure if I am permitted to share WRDS, CRSP, or Sustainalytics data on a 
public GitHub repository, so erring on the side of caution I have not posted it
here. However, a detailed decription of the data collection and cleaning is 
provided both here and in the main body of the thesis.

Sections of the code will have to be rewitten depending on which platform
the reader is on, and to change directories and paths used to save/load data.