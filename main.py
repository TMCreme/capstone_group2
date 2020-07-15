import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import math


start_date = '2006-06-01'
end_date = '2020-05-31'

joburg_df = yf.download('JSE.JO', start=start_date, end=end_date)
snp500_df = yf.download('^GSPC', start=start_date, end=end_date)
shanghai_df = yf.download('000001.SS', start=start_date, end=end_date)

# Calculating returns 
joburg_df["Returns"] = joburg_df["Adj Close"].pct_change()
snp500_df["Returns"] = snp500_df["Adj Close"].pct_change()
shanghai_df["Returns"] = shanghai_df["Adj Close"].pct_change()

# Describing the basic statistics of the data
print(joburg_df.describe())

## Plotting the returns data 
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(joburg_df["Returns"])
ax1.plot(snp500_df["Returns"])
ax1.plot(shanghai_df["Returns"])
ax1.set_xlabel("Date")
ax1.set_ylabel("Returns")
ax1.set_title("Stock Index daily returns for the dataset")
plt.show()

# dw_stat = ts.durbin_watson(joburg_df["Returns"])

# LJung Box Test for auto correlation on the JSE.JO
joburg_df.dropna(subset = ["Returns"], inplace=True)
res = sm.tsa.ARMA(joburg_df["Returns"], (1,1)).fit(disp=-1)
lbox_test = acorr_ljungbox(res.resid,lags=2, return_df=True)

print(lbox_test)



