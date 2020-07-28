#!/usr/bin/env python
# coding: utf-8

# # Capstone Draft


#Import libraries
import numpy as np
from numpy import percentile
import pandas as pd
from pandas import Series, DataFrame
from pandas.plotting import register_matplotlib_converters
import matplotlib
import matplotlib.pyplot as plt
register_matplotlib_converters()
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest, anderson
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.iolib.summary2 import summary_col
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from arch import arch_model


#change the directory here to where your data is. 
address = 'AdjustedClose_data.xlsx'

#Import the data from csv file and store into an object called assets.
assets = pd.read_excel(address,  parse_dates=True, squeeze=True)

print(assets.head())

#Remove missing values from assets object and store in index_data object.
index_data = assets.dropna()

print(index_data.head())

print(assets.tail())

# # Statistical Properties of Price Indices ( S&P500, JSE.JO, 000001.SS)

# Function to plot stock prices using different date ranges
def plotting_hstorical_prices_with_periods(from_date="2006-06-01", to_date="2020-05-29"):
    # get_ipython().run_line_magic('matplotlib', 'inline')
    matplotlib.rcParams['figure.figsize'] = [15, 7]
    sliced_assets = assets[assets.Date >=from_date]
    sliced_assets = sliced_assets[sliced_assets.Date <=to_date]
    plt.plot(sliced_assets['Date'], sliced_assets['JSE Adj Close'], label='JSE Adj Close')
    plt.plot(sliced_assets['Date'], sliced_assets['SS Adj Close'], label='SS Adj Close')
    plt.plot(sliced_assets['Date'], sliced_assets['GSPC Adj Close'], label='GSPC Adj Close')
    plt.xlabel('Date')
    plt.ylabel('Index')
    plt.legend()
    plt.grid()
    plt.show()


#Plot the historical evolution of index prices from June 2006 to May 2020. 
plotting_hstorical_prices_with_periods()

#Plot the historical evolution of index prices from July 2007 to May 2009. == 2008 Financial Crisis
plotting_hstorical_prices_with_periods("7/1/2007","6/30/2009")

#Plot the historical evolution of index prices from July 2009 – Nov. 2019. 
plotting_hstorical_prices_with_periods("6/1/2009", "11/30/2019")

#Plot the historical evolution of index prices from Dec. 2019 – May 2020. 
plotting_hstorical_prices_with_periods("12/1/2019", "5/31/2020")

#Separate the indices into individual components into individual objects
jsejo = index_data['JSE Adj Close']
snp500 = index_data['GSPC Adj Close']
shan = index_data['SS Adj Close']


# In[11]:


#Plot historical evolution of S&P 500 from June 2006 to May 2020.
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(snp500)
plt.ylabel('Price')
plt.title('Historical Stock Index of the S&P 500 Index')
plt.xlabel('Date')
plt.grid()


#Plot the historical evoluation of Johannesburg Stock Index from June 2006 to May 2020. 
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(jsejo)
plt.ylabel('Price')
plt.title('Historical Stock Index of the Johannesburg Stock Index')
plt.xlabel('Date')
plt.grid()

#Plot the historical evolution of Shanghai Index from June 2006 to May 2020. 
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(shan)
plt.ylabel('Price')
plt.title('Historical Stock Index of the Shanghai Index')
plt.xlabel('Date')
plt.grid()

#Calculate sample statistics of Indices (mean, std. dev, skewness, kurtosis)
means = index_data.mean()
stddevs = index_data.std()
skewness = index_data.skew()
kurtosis = index_data.kurt()

print('Mean of Indices \n\n', means, "\n\n")
print('Standard Deviation of Indices \n\n', stddevs, "\n\n")
print('Skewness of Indices \n\n', skewness, "\n\n")
print('Kurtosis of Indices \n\n', kurtosis, "\n\n")

#Normalize the various indices
normalised_prices = (index_data.set_index("Date") - means)/stddevs

# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [18, 7]
plt.plot(normalised_prices)  
plt.ylabel('Normalised Indices')  
plt.legend(assets)  
plt.grid()


# #  Statistical Properties of Returns and Volatilities

#Plot log returns
log_returns = np.log(index_data.set_index('Date')).diff().dropna()

matplotlib.rcParams['figure.figsize'] = [18, 7]
plt.plot(log_returns)
plt.ylabel('Log returns')
plt.legend(assets)
plt.grid()


#Plot Shanghai Index Returns
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(log_returns['SS Adj Close'])  
plt.ylabel('Returns')  
plt.title('Return Series of the Shanghai Index')
plt.xlabel('Date')  
plt.grid()



#Plot Johannesburg Index Returns
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(log_returns['JSE Adj Close'])  
plt.ylabel('Returns')  
plt.title('Return Series of the Johannesburg Stock Index')
plt.xlabel('Date')  
plt.grid()

#Plot S&P500 Index Returns
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(log_returns['GSPC Adj Close'])  
plt.ylabel('Returns')  
plt.title('Return Series of the S&P500 Stock Index')
plt.xlabel('Date')  
plt.grid()

#Obtain mean, std. deviation, statistical moments of indicies.

return_mean = log_returns.mean()
return_stddev = log_returns.std()
return_skewness = log_returns.skew()
return_kurt = log_returns.kurt()

print('Mean of Return Indices \n\n', return_mean, "\n\n")
print('Standard Deviation of Return Indices \n\n', return_stddev, "\n\n")
print('Skewness of Return Indices \n\n', return_skewness, "\n\n")
print('Kurtosis of Return Indices \n\n', return_kurt, "\n\n")

#5 Point Summary of Market Returns
quart_Shanghai = percentile(log_returns['SS Adj Close'], [0, 25, 50, 75, 100])
quart_JSE = percentile(log_returns['JSE Adj Close'], [0, 25, 50, 75, 100])
quart_SNP500 = percentile(log_returns['GSPC Adj Close'], [0, 25, 50, 75, 100])
 
heading : ['Min', '25th Percentile', 'Median', '75th Percentile', 'Max']

percent = {'Asset' : ['S&P 500' , 'JSE Limited', 'Shanghai Stock Exchange'], 
           'Minimum' : [quart_SNP500[0], quart_JSE[0], quart_Shanghai[0]],
           '25th Percentile':[quart_SNP500[1], quart_JSE[1], quart_Shanghai[1]],
           'Median' : [quart_SNP500[2], quart_JSE[2], quart_Shanghai[2]],
           '75th Percentile' : [quart_SNP500[3], quart_JSE[3], quart_Shanghai[3]],
           'Maximum' : [quart_SNP500[4], quart_JSE[4], quart_Shanghai[4]]}

percentile_dataframe = pd.DataFrame(percent, columns = ['Asset', 'Minimum', '25th Percentile', 'Median', '75th Percentile', 'Maximum'])
print(percentile_dataframe)

#Obtain the number of observations in index_data object 
number_of_rows = len(index_data)
print('Today number of trading days from Jun 2006 to May 2020:', number_of_rows)

#14 years counting from June 2006 to May 2020
Average_trading_days_per_year = number_of_rows/14
print('Average trading days:', Average_trading_days_per_year)

#obtain the tail observations
print(index_data.tail())


return_mean = log_returns.mean()
return_stddev = log_returns.std()
return_annualized_volatility = return_stddev*np.sqrt(Average_trading_days_per_year)
return_risk_ratio = return_mean / return_stddev

print('Annualized volatility of Return Indices \n\n', return_annualized_volatility, '\n\n')
print('Return/Risk Ratio of the Return Indices \n\n', return_risk_ratio, '\n\n')


# # Normality Tests of Returns

#Normality Test1: Jarque Bera Test
print('Jarque-Bera Normality test of returns')
print("Ticker \t\t\t Result")
for ticker in assets.set_index('Date'):
   print(ticker + '\t\t ', jarque_bera(log_returns[ticker]))


#Normality Test2: Shapiro-Wilk Test
print('Shapiro-Wilk Normality of returns test')
print("Ticker \t\t\t Result")
for ticker in assets.set_index('Date'):
   print(ticker + ' \t\t ', shapiro(log_returns[ticker]))


#Normality Test3: D^Agostino's K^2 Test
print("D^Agostino's K^2 Normality of returns test")
print("Ticker \t\t\t Result")
for ticker in assets.set_index('Date'):
   print(ticker + '\t\t ', normaltest(log_returns[ticker]))


#Normality Test4: Anderson Test

def anderson_normality_test(ticker):
    ticker_dict = {"GSPC":"S&P 500", "JSE":"JSE Limited","SS":"Shanghai SE"}
    print("Anderson Test of Normality of returns for ", ticker_dict[ticker])
    result = anderson(log_returns[ticker+' Adj Close'])
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject HO)' % (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject HO)' % (sl, cv))


anderson_normality_test('GSPC')

anderson_normality_test('JSE')

anderson_normality_test('SS')


#Histogram Daily JSE.JO
n, bins, patches = plt.hist(log_returns['JSE Adj Close'], 100, density=True)

plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Histogram of daily returns of Johannesburg Stock Exchange')
plt.grid(True)
plt.show()


#Histogram Shanghai Stock
n, bins, patches = plt.hist(log_returns['SS Adj Close'], 100, density=True)

plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Histogram of daily returns of Shanghai Stock Exchange')
plt.grid(True)
plt.show()


#Histogram S&P 500
n, bins, patches = plt.hist(log_returns['GSPC Adj Close'], 100, density=True)

plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Histogram of daily returns of S&P 500')
plt.grid(True)
plt.show()

#QQPlot of Johannesburg Stock
# get_ipython().run_line_magic('matplotlib', 'inline')

qqplot(log_returns['JSE Adj Close'] , line='s')
plt.title('QQ Plot of Johannesburg Stock')
plt.grid(False)
plt.show()


#QQPlot of Shanghai Stock
# get_ipython().run_line_magic('matplotlib', 'inline')

qqplot(log_returns['SS Adj Close'] , line='s')
plt.title('QQ Plot of Shanghai Stock')
plt.grid(False)
plt.show()



#QQPlot of S&P500 Stock
# get_ipython().run_line_magic('matplotlib', 'inline')

qqplot(log_returns['GSPC Adj Close'] , line='s')
plt.title('QQ Plot of S&P500 Index')
plt.grid(False)
plt.show()


# Function to calculate Correlation of the various return indices using different date ranges
def caculate_correlation_of_log_returns_daily(from_date="2006-06-01", to_date="2020-05-29"):
    reset_log_returns = log_returns.reset_index()
    print("Correlations among the various returns of the Daily Stock from "+ from_date+ " to "+to_date)
    sliced_returns = reset_log_returns[reset_log_returns.Date >=from_date]
    sliced_returns = sliced_returns[reset_log_returns.Date <=to_date]
    return sliced_returns.corr()

def calculate_correlation_of_price_indices(from_date="2006-06-01", to_date="2020-05-29"):
    #Correlation of the various price indices
    print("Correlations among the various price indices from "+from_date +" to "+to_date)
    sliced_index = index_data[index_data.Date >=from_date]
    sliced_index = sliced_index[index_data.Date <=to_date]
    return sliced_index.corr()

# Function to calculate Correlation of the various return indices using different date ranges
# Full Period: July 2006 – June 2020
caculate_correlation_of_log_returns_daily()

# Function to calculate Correlation of the various return indices using different date ranges
# Full Period: July 2006 – June 2020
calculate_correlation_of_price_indices()

# Function to calculate Correlation of the various return indices using different date ranges
# 2008 Financial Crisis Period: July 2007 – June 2009
caculate_correlation_of_log_returns_daily("2007-07-01", "2009-06-30")

# Function to calculate Correlation of the various return indices using different date ranges
# 2008 Financial Crisis Period: July 2007 – June 2009
calculate_correlation_of_price_indices("2007-07-01", "2009-06-30")

# Function to calculate Correlation of the various return indices using different date ranges
# July 2009 – Nov. 2019
caculate_correlation_of_log_returns_daily("2009-07-01", "2019-11-30")

# Function to calculate Correlation of the various return indices using different date ranges
# July 2009 – Nov. 2019
calculate_correlation_of_price_indices("2009-07-01", "2019-11-30")

# Function to calculate Correlation of the various return indices using different date ranges
# Covid-19 Period: Dec. 2019 – May 2020
caculate_correlation_of_log_returns_daily("2019-12-01", "2020-05-31")

# Function to calculate Correlation of the various return indices using different date ranges
# Covid-19 Period: Dec. 2019 – May 2020
calculate_correlation_of_price_indices("2019-12-01", "2020-05-31")


#Correlation of normalized prices
log_normalised_returns = np.log(normalised_prices).diff().dropna()

#Correlation of the various normalized price indices
print("Correlations among the various normalised price indices \n\n", log_normalised_returns.corr() , "\n\n")




#Density Plot of Johannesburg Stock Returns
plt.figure(1)
plt.subplot(211)
plt.title('Density Plot of Johannesburg Stock Returns')
log_returns['JSE Adj Close'].hist()
plt.subplot(212)
log_returns['JSE Adj Close'].plot(kind='kde')
plt.show()


#Density Plot of Shanghai Returns
plt.figure(1)
plt.subplot(211)
plt.title('Density Plot of Shanghai Stock Returns')
log_returns['SS Adj Close'].hist()
plt.subplot(212)
log_returns['SS Adj Close'].plot(kind='kde')
plt.show()


#Density Plot of S&P500 Returns
plt.figure(1)
plt.subplot(211)
plt.title('Density Plot of S&P500 Stock Returns')
log_returns['GSPC Adj Close'].hist()
plt.subplot(212)
log_returns['GSPC Adj Close'].plot(kind='kde')
plt.show()


# # Autocorrelation of Returns

#Autocorrelation Analysis of Index

#ACF
plot_acf(jsejo, title="Johannesburg Stock ACF", lags=25)
plot_acf(snp500, title="S&P 500 Index ACF", lags=25)
plot_acf(shan, title="Shanghai Stock Index ACF", lags=25)
plt.show()


#ACF Squared
plot_acf(jsejo**2, title="Johannesburg Stock ACF Squared", lags=25)
plot_acf(snp500**2, title="S&P 500 Index ACF Squared", lags=25)
plot_acf(shan**2, title="Shanghai Stock Index ACF Squared", lags=25)
plt.show()



#PACF Plots
plot_pacf(jsejo, title="Johannesburg Stock PACF", lags=25)
plot_pacf(snp500, title="S&P 500 Index PACF", lags=25)
plot_pacf(shan, title="Shanghai Stock Index PACF", lags=25)
plt.show()


#PACF Squared Plots
plot_pacf(jsejo**2, title="Johannesburg Stock PACF Squared", lags=25)
plot_pacf(snp500**2, title="S&P 500 Index PACF Squared", lags=25)
plot_pacf(shan**2, title="Shanghai Stock Index PACF Squared", lags=25)
plt.show()


# # Stationary Tests of Returns


#Assessing Stationarity of Returns

def adfuller_stationarity(ticker):
    ticker_dict = {"GSPC":"S&P 500", "JSE":"JSE Limited","SS":"Shanghai SE"}
    print("Augmented Dickey Fuller Test for ", ticker_dict[ticker])
    # ADF Test
    ticker_stationarity = adfuller(log_returns[ticker+' Adj Close'])
    print(f'ADF Statistic: \t\t {ticker_stationarity[0]}')
    print(f'p-value: \t\t {ticker_stationarity[1]}')
    for key, value in ticker_stationarity[4].items():
        print(f'Critial Values at : \t {key} \t\t {value}')
        
def kpss_stationarity_test(ticker):
    ticker_dict = {"GSPC":"S&P 500", "JSE":"JSE Limited","SS":"Shanghai SE"}
    print("KPSS Stationarity Test for ", ticker_dict[ticker])
    # KPSS Test
    ticker_stationarity = kpss(log_returns[ticker +' Adj Close'])
    print('\nKPSS Statistic: \t\t %f' % ticker_stationarity[0])
    print('p-value: \t\t\t %f' % ticker_stationarity[1])
    for key, value in ticker_stationarity[3].items():
        print(f'Critial Values at : \t   {key} \t\t {value}')


adfuller_stationarity('GSPC')

kpss_stationarity_test('GSPC')

adfuller_stationarity('JSE')


kpss_stationarity_test('JSE')

adfuller_stationarity('SS')

kpss_stationarity_test('SS')


# # Volatility Modelling of Market Returns

#Volatility Modelling of Market Returns
#JSE return ARCH(1)

amjse1 = arch_model(100*log_returns['JSE Adj Close'], p=1, q=0 )
resamjse1 = amjse1.fit()
print(resamjse1.summary())


#Volatility Modelling of Market Returns
#GSPC return ARCH(1)

amsnp1 = arch_model(100*log_returns['GSPC Adj Close'], p=1, q=0 )
resamsnp1 = amsnp1.fit()
print(resamsnp1.summary())


#Volatility Modelling of Market Returns
#SSE return ARCH(1)

amsse1 = arch_model(100*log_returns['SS Adj Close'], p=1, q=0 )
resamsse1 = amsse1.fit()
print(resamsse1.summary())



#Volatility Modelling of Market Returns
#JSE return ARCH(1)

amjse2 = arch_model(100*log_returns['JSE Adj Close'], p=2, q=0 )
resamjse2 = amjse2.fit()
print(resamjse2.summary())



#Volatility Modelling of Market Returns
#GSPC return ARCH(2)

amsnp2 = arch_model(100*log_returns['GSPC Adj Close'], p=2, q=0 )
resamsnp2 = amsnp2.fit()
print(resamsnp2.summary())


#Volatility Modelling of Market Returns
#SSE return ARCH(2)

amsse2 = arch_model(100*log_returns['SS Adj Close'], p=2, q=0 )
resamsse2 = amsse2.fit()
print(resamsse2.summary())


#Volatility Modelling of Market Returns
#JSE return ARCH(3)

amjse3 = arch_model(100*log_returns['JSE Adj Close'], p=3, q=0 )
resamjse3 = amjse3.fit()
print(resamjse3.summary())



#Volatility Modelling of Market Returns
#GSPC return ARCH(3)

amsnp3 = arch_model(100*log_returns['GSPC Adj Close'], p=3, q=0 )
resamsnp3 = amsnp3.fit()
print(resamsnp3.summary())


#Volatility Modelling of Market Returns
#SSE return ARCH(3)

amsse3 = arch_model(100*log_returns['SS Adj Close'], p=3, q=0 )
resamsse3 = amsse3.fit()
print(resamsse3.summary())


#Volatility Modelling of Market Returns
#JSE return ARCH(4)

amjse4 = arch_model(100*log_returns['JSE Adj Close'], p=4, q=0 )
resamjse4 = amjse4.fit()
print(resamjse4.summary())


#Volatility Modelling of Market Returns
#GSPC return ARCH(4)

amsnp4 = arch_model(100*log_returns['GSPC Adj Close'], p=4, q=0 )
resamsnp4 = amsnp4.fit()
print(resamsnp4.summary())


#Volatility Modelling of Market Returns
#SSE return ARCH(4)

amsse4 = arch_model(100*log_returns['SS Adj Close'], p=4, q=0 )
resamsse4 = amsse4.fit()
print(resamsse4.summary())


#Volatility Modelling of Market Returns
#JSE return GARCH(1,1)

amjse11 = arch_model(100*log_returns['JSE Adj Close'], p=1, q=1 )
resamjse11 = amjse11.fit()
print(resamjse11.summary())


#Volatility Modelling of Market Returns
#GSPC return GARCH(1,1)

amsnp11 = arch_model(100*log_returns['GSPC Adj Close'], p=1, q=1 )
resamsnp11 = amsnp11.fit()
print(resamsnp11.summary())


#Volatility Modelling of Market Returns
#SSE return GARCH(1,1)

amsse11 = arch_model(100*log_returns['SS Adj Close'], p=1, q=1 )
resamsse11 = amsse11.fit()
print(resamsse11.summary())


# # Long Run and Short Run Relationship between GSPC and the emerging markets

#Granger Causality Test btn GSPC and JSE to ascertain short term relationship

grangercausalitytests(index_data[['GSPC Adj Close', 'JSE Adj Close']], maxlag=2)

#Granger Causality Test btn GSPC and JSE to ascertain short term relationship

grangercausalitytests(index_data[['GSPC Adj Close', 'SS Adj Close']], maxlag=2)

#Return Causality

grangercausalitytests(log_returns[['GSPC Adj Close', 'JSE Adj Close']], maxlag=2)


#Return Causality

grangercausalitytests(log_returns[['GSPC Adj Close', 'SS Adj Close']], maxlag=2)


#Finding comovement/long term relationships between JSEJO and SNP500
#STEPS
#1. Regress SNP500 on JSEJO
#2. Estimate Residuals (Long run relationship) 
#3. Test for Stationarity( Stationarity implies long term relationship)


#STEP 1
reg1 = sm.OLS(endog=snp500, exog=jsejo, missing='drop')
results = reg1.fit()
print(results.summary())


#2. Estimate Residuals (Long run relationship)
#Plot the long-term relationship between SNP500 and JSE.JO stock
longrrunrela_jse = results.resid
plt.plot(longrrunrela_jse)
plt.show()



#3. Test for Stationarity( Stationarity implies long term relationship or equilibrium)
#Checking the cointegration relationship using dickey fuller test of residual. 
dickfullerjse_result = adfuller(longrrunrela_jse)
print('ADF Statistic: %f' % dickfullerjse_result[0])
print('p-value: %f' % dickfullerjse_result[1])
print('Critical Values:')
for key, value in dickfullerjse_result[4].items():
    print('\t%s: %.3f' % (key, value))


#Comovement relationship between SNP500 and Shanghai Stock Index

reg2 = sm.OLS(endog=snp500, exog=shan, missing='drop')
results2 = reg2.fit()
print(results2.summary())


#Plot the long-term relationship between SNP500 and SHANGHAI stock
longrrunrela_shan = results2.resid
plt.plot(longrrunrela_shan)
plt.show()


#Checking the cointegration relationship using dickey fuller test of residual. 
dickfullershan_result = adfuller(longrrunrela_shan)
print('ADF Statistic: %f' % dickfullershan_result[0])
print('p-value: %f' % dickfullershan_result[1])
print('Critical Values:')
for key, value in dickfullershan_result[4].items():
    print('\t%s: %.3f' % (key, value))


# Regime Switching model
#Fit the model with stock returns for JSE.JO
markov_model_jse = sm.tsa.MarkovRegression(log_returns["JSE Adj Close"].dropna(), k_regimes=3, 
                                       trend='nc', switching_variance=True)
markov_results_jse = markov_model_jse.fit()
markov_results_jse.summary()



#Plot JSE Smoothed marginal Probabilities of the Switiching model for JSE  == low-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_jse.smoothed_marginal_probabilities[0])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a low-variance regime for JSE stock returns')
plt.xlabel('Date')  
plt.grid()




#Plot JSE Smoothed marginal Probabilities of the Switiching model for JSE  == medium-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_jse.smoothed_marginal_probabilities[1])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a medium-variance regime for JSE stock returns')
plt.xlabel('Date')  
plt.grid()


#Plot JSE Smoothed marginal Probabilities of the Switiching model for JSE  == high-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_jse.smoothed_marginal_probabilities[2])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a high-variance regime for JSE stock returns')
plt.xlabel('Date')  
plt.grid()



# Regime Switching model
#Fit the model with stock returns for S&P 500
markov_model_snp = sm.tsa.MarkovRegression(log_returns["GSPC Adj Close"].dropna(), k_regimes=3, 
                                       trend='nc', switching_variance=True)
markov_results_snp = markov_model_snp.fit()
markov_results_snp.summary()


#Plot JSE Smoothed marginal Probabilities of the Switiching model for GSPC  == low-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_snp.smoothed_marginal_probabilities[0])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a low-variance regime for S&P 500 stock returns')
plt.xlabel('Date')  
plt.grid()



#Plot JSE Smoothed marginal Probabilities of the Switiching model for GSPC  == medium-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_snp.smoothed_marginal_probabilities[1])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a medium-variance regime for S&P 500 stock returns')
plt.xlabel('Date')  
plt.grid()


#Plot JSE Smoothed marginal Probabilities of the Switiching model for GSPC  == high-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_snp.smoothed_marginal_probabilities[2])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a high-variance regime for S&P 500 stock returns')
plt.xlabel('Date')  
plt.grid()


# Regime Switching model
#Fit the model with stock returns for the SSE
markov_model_sse = sm.tsa.MarkovRegression(log_returns["SS Adj Close"].dropna(), k_regimes=3, 
                                       trend='nc', switching_variance=True)
markov_results_sse = markov_model_sse.fit()
markov_results_sse.summary()


#Plot JSE Smoothed marginal Probabilities of the Switiching model for SSE  == low-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_sse.smoothed_marginal_probabilities[0])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a low-variance regime for Shaghai SE stock returns')
plt.xlabel('Date')  
plt.grid()


#Plot JSE Smoothed marginal Probabilities of the Switiching model for SSE  == medium-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_sse.smoothed_marginal_probabilities[1])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a medium-variance regime for Shaghai SE stock returns')
plt.xlabel('Date')  
plt.grid()



#Plot JSE Smoothed marginal Probabilities of the Switiching model for SSE  == high-variance
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15, 7]
plt.plot(markov_results_sse.smoothed_marginal_probabilities[2])  
plt.ylabel('Returns')  
plt.title('Smoothed probability of a low-variance regime for Shaghai SE stock returns')
plt.xlabel('Date')  
plt.grid()