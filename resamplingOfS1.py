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
s1=pd.read_csv(".\HistoricalData_AMZN.csv",header=0, index_col="Date", parse_dates=True)
# resampling of data
# Zooming into data set
# Zooming in to see data for the year 2021 to current day
s1_21=s1.loc[("2021-01-01"):("2021-06-09")]
plt.plot(s1_21.index, s1_21["Close/Last"])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
#plt.show()
plt.savefig("zooming.png")
 # Resampling(Quarterly)
monthly_s1_21=s1_21.resample("4M").mean()
plt.scatter(monthly_s1_21.index,monthly_s1_21["Close/Last"])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
#plt.show()
plt.savefig("resamplingQuarterly.png")
# Resampling (Weekly)
weekly_s1_21=s1_21.resample("W").mean()
print(weekly_s1_21.head())
plt.plot(weekly_s1_21.index,weekly_s1_21["Close/Last"],"-0")
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
#plt.show()
plt.savefig("resamplingWeekly.png")









