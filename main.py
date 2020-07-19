import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.regression.linear_model import OLS
from scipy import stats 
import numpy as np
import math

# Date ranges for the data to be used
start_date = '2006-06-01'
end_date = '2020-05-31'

# Daily data
joburg_daily_df = yf.download('JSE.JO', start=start_date, end=end_date)
snp500_daily_df = yf.download('^GSPC', start=start_date, end=end_date)
shanghai_daily_df = yf.download('000001.SS', start=start_date, end=end_date)

# Monthly data
joburg_monthly_df = yf.download('JSE.JO', start=start_date, end=end_date, interval="1mo")
snp500_monthly_df = yf.download('^GSPC', start=start_date, end=end_date, interval="1mo")
shanghai_monthly_df = yf.download('000001.SS', start=start_date, end=end_date, interval="1mo")

# Calculating daily returns 
joburg_daily_df["Returns"] = joburg_daily_df["Adj Close"].pct_change().dropna()
snp500_daily_df["Returns"] = snp500_daily_df["Adj Close"].pct_change().dropna()
shanghai_daily_df["Returns"] = shanghai_daily_df["Adj Close"].pct_change().dropna()

# Calculating monthly returns 
joburg_monthly_df["Returns"] = joburg_monthly_df["Adj Close"].pct_change().dropna()
snp500_monthly_df["Returns"] = snp500_monthly_df["Adj Close"].pct_change().dropna()
shanghai_monthly_df["Returns"] = shanghai_monthly_df["Adj Close"].pct_change().dropna()

# Describing the basic statistics of the data
print("JSE.JO Daily Stock Return Basic Statistics Description \n\n", joburg_daily_df.describe() + "\n\n")
print("S%P500 Daily Stock Return Basic Statistics Description \n\n", joburg_daily_df.describe() + "\n\n")
print("Shanghai Daliy Stock Return Basic Statistics Description \n\n", joburg_daily_df.describe() + "\n\n")

## Plotting the Daily returns data 
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
# ax1.plot(joburg_daily_df["Returns"])
ax1.plot(snp500_daily_df["Returns"])
ax1.plot(shanghai_daily_df["Returns"])
ax1.set_xlabel("Date")
ax1.set_ylabel("Returns")
ax1.set_title("Stock Index daily returns for the dataset")
plt.show()

## Plotting the Monthly returns data 
fig = plt.figure(figsize=(50,50))
ax2 = fig.add_axes([0.1,0.1,0.8,0.8])
ax2.plot(joburg_monthly_df["Returns"], label="JSE.JO")
ax2.plot(snp500_monthly_df["Returns"], label="S&P 500")
ax2.plot(shanghai_monthly_df["Returns"], label="Shanghai")
ax2.set_xlabel("Date")
ax2.set_ylabel("Returns")
ax2.set_title("Stock Index Monthly returns for the dataset")
ax2.legend()
plt.show()


# Correlation between the returns for the 3 indices
daily_returns_df = pd.DataFrame({"JSE.JO":joburg_daily_df["Returns"],
                        "S&P500":snp500_daily_df["Returns"],"Shanghai":shanghai_daily_df["Returns"]})

monthly_returns_df = pd.DataFrame({"JSE.JO":joburg_monthly_df["Returns"],
                        "S&P500":snp500_monthly_df["Returns"],"Shanghai":shanghai_monthly_df["Returns"]})

print("Correlations between the various returns of the Daily Stock")
print(daily_returns_df.corr())
print("Correlations between the various returns of the Monthly Stock data")
print(monthly_returns_df.corr())

# Fitting the data into an OLS regression model
joburg_model = sm.OLS(daily_returns_df['JSE.JO'], np.ones(len(daily_returns_df['JSE.JO'])))
joburg_results = joburg_model.fit()
# The summary presents results on Durbin watson, Jarques-Bera, P-val, etc
print(joburg_results.summary())

# Tests of heteroscedasticity (Breusch-Pagan Test)
joburg_breush_pagan_test = het_breuschpagan(joburg_results.resid, joburg_results.model.exog)
print("JSE.JO Breusch-Pagan Test for heteroscedasticity: \n\n", joburg_breush_pagan_test)

# Serial Correlation Test (Ljung-Box test)
joburg_serial_corr = acorr_ljungbox(joburg_results.resid,lags=2, return_df=True)
print("JSE.JO Ljung-Box Test for Serial Correlation: \n\n", joburg_serial_corr)


# ACF and PACF Plots
# JSE.JO
plot_acf(joburg_daily_df["Adj Close"], title="JSE.JO Auto Correlation for daily Prices")
plot_pacf(joburg_daily_df["Adj Close"],title="JSE.JO Partial AutoCorrelation for daily Prices")

# S&P500
plot_acf(snp500_daily_df["Adj Close"], title="S&P500 Auto Correlation for daily Prices")
plot_pacf(snp500_daily_df["Adj Close"],title="S&P500 Partial AutoCorrelation for daily Prices")

# Shanghai
plot_acf(shanghai_daily_df["Adj Close"], title="Shanghai Auto Correlation for daily Prices")
plot_pacf(shanghai_daily_df["Adj Close"],title="Shanghai Partial AutoCorrelation for daily Prices")


# Test of Stationarity (Augmented Dickey-Fuller (ADF))
print("============AUGMENTED DICKEY-FULLER TEST FOR STATIONARITY=============")
test_keys = ['Test Statistic','p-value','#Lags Used','Number of Observations']
# JSE.JO
print("JSE.JO RESULTS ")
joburg_adf_test = ts.adfuller(joburg_daily_df['Adj Close'])
for i in test_keys:
    print(i +" : " + str(joburg_adf_test[test_keys.index(i)]))
for key, value in joburg_adf_test[4].items():
    print(key + " Critical Value : " + str(value))
print("\n\n")

# S&P500
print("S&P500 RESULTS ")
snp500_adf_test = ts.adfuller(snp500_daily_df['Adj Close'])
for i in test_keys:
    print(i +" : " + str(snp500_adf_test[test_keys.index(i)]))
for key, value in snp500_adf_test[4].items():
    print(key + " Critical Value : " + str(value))
print("\n\n")

# Shanghai
print("Shanghai RESULTS ")
shanghai_adf_test = ts.adfuller(shanghai_daily_df['Adj Close'])
for i in test_keys:
    print(i +" : " + str(shanghai_adf_test[test_keys.index(i)]))
for key, value in shanghai_adf_test[4].items():
    print(key + " Critical Value : " + str(value))
print("\n\n")



# Fitting the data into an ARMA model 
# Haven't explored to see which is the best
joburg_daily_df.dropna(subset = ["Returns"], inplace=True)
joburg_model2 = sm.tsa.ARMA(joburg_daily_df["Returns"], (2,2,2)).fit(disp=-1)
print(joburg_model2.summary())








