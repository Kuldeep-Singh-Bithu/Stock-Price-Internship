# measurement of the change in variance in the returns of stock over specific period of time
# Higher the volatility Higher the risk
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# reading CSV file
s1=pd.read_csv(".\HistoricalData_AMZN.csv")
# seting date as a index
s1_2=pd.read_csv(".\HistoricalData_AMZN.csv",header=0, index_col="Date",parse_dates=True)
# Moving windows
# to analyse long term trends
# with respect to Daily percentages
daily_close_s1=s1_2[["Close/Last"]]
# daily returns
# pct means percentage
daily_pct_change_s1=daily_close_s1.pct_change()
# replacing NA values with 0
daily_pct_change_s1.fillna(0,inplace=True)
print(daily_pct_change_s1.head())

min_periods=75
# Calculating the Volatility
# Calculating variance of these dail percentage values
vol=daily_pct_change_s1.rolling(min_periods).std()*np.sqrt(min_periods)
vol.fillna(0,inplace=True)
print(vol.tail())
# Plotting the Volatility
vol.plot(figsize=(10,8))
#plt.show()
plt.savefig("VolatilityGraph.png")
# Rolling means(Trend and Seasonality)
s1_close_px=s1_2["Close/Last"]
# Short term moving window rolling mean
s1_2["42"]=s1_close_px.rolling(window=40).mean()
# Long term moving window rolling mean
s1_2["252"]=s1_close_px.rolling(window=252).mean()
# plot the  closing price,the short and long terms windows of rolling means
print(s1_2.head())
s1_2[["Close/Last","42","252"]].plot(title="Amazon")
#plt.show()
plt.savefig("TrendAndSeasonality.png")




