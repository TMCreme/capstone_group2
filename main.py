import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt 
import math


start_date = '2008-01-01'
end_date = '2019-12-31'
country_data = yf.Ticker('^JN0U.JO')
df = country_data.history(start=start_date, end=end_date)
# df['Close'].plot(title="TSLA's stock price")
print(df.head())

data_brazil = yf.Ticker('^bvsp')
df1 = data_brazil.history(start=start_date, end=end_date)
print(df1.head())
