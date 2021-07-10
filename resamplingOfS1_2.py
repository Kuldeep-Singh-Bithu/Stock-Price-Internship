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
# resampling of data
# Analysing Difference between Levels(Resampling Weekly)
# Difference Between Open value and close value
# this will give better pitcher for examine the company whether profit or lose
s1_2["diff"]=s1_2["Open"]-s1_2["Close/Last"]
s1_diff=s1_2.resample("W").mean()
plt.scatter(s1_diff.loc["2021-01-01":"2021-06-09"].index, s1_diff.loc["2021-01-01":"2021-06-09"]["diff"])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
#plt.show()
plt.savefig("DifferenceBetweenOpenCloseResamplingWeekly.png")


