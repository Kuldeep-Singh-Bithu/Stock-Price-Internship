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
s1=pd.read_csv(".\HistoricalData_AMZN.csv",header=0,index_col="Date",parse_dates=True)
# understanding the data
#s1.head()
print(s1.head(20))
print(s1.tail())
print(s1.describe())
# cleaning data
missing_val=s1.isnull().sum()
print(missing_val)
# there is no missing value or missing data
# checking for unique value
print(s1.nunique())
# Time serise Data
plt.plot(s1.index, s1["Close/Last"])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.gca().xaxis.set_major_locator(mdates.YearLocator()) # formatting data for plotting
plt.grid(True) # to read data more clearly on the graph
plt.xticks(rotation=90) # rotating dates to 90 degree so that we can read more clearly
#plt.show()
plt.savefig("ClosePriceOnGraphEDA.png")






