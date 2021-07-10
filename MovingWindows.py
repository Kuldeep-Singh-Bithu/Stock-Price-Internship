#Amazone stock price prediction
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
# lets plot the histogram
daily_pct_change_s1.hist(bins=50)
#plt.show()
plt.savefig("percentageChangeHistogram")






