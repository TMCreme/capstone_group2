import pandas as pd 
from pandas_datareader import data
import yfinance as yf
import matplotlib.pyplot as plt 
import math


start_date = '2008-01-01'
end_date = '2019-12-31'
country_data = yf.Ticker('^JN0U.JO')
df = country_data.history(start=start_date, end=end_date)
# df['Close'].plot(title="TSLA's stock price")
print(df.head())

