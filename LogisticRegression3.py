import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pickle

from statsmodels.formula.api import logit

# Techinical Indicators
import talib as ta

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# reading CSV file
df=pd.read_csv(".\HistoricalData_AMZN.csv")

# seting date as a index
#df=pd.read_csv(".\HistoricalData_AMZN.csv",header=0,index_col="Date",parse_dates=True)
df=pd.read_csv(".\HistoricalData_AMZN.csv")
# understanding the data
print(df.head())
# set the date as the index for hte data
df=df.set_index(pd.DatetimeIndex(df['Date'].values))
#Give the index a name
df.index.name="Date"
#Show the data
print(df.head())

# Removing $ sign from data set
df['Close/Last'] = df['Close/Last'].apply(lambda x : str(x).strip('$') if '$' in str(x) else str(x))
df['Open'] = df['Open'].apply(lambda x : str(x).strip('$') if '$' in str(x) else str(x))
df['High'] = df['High'].apply(lambda x : str(x).strip('$') if '$' in str(x) else str(x))
df['Low'] = df['Low'].apply(lambda x : str(x).strip('$') if '$' in str(x) else str(x))

df['Close/Last'] = df['Close/Last'].apply(lambda x : float(x))
df['Open'] = df['Open'].apply(lambda x : float(x))
df['High'] = df['High'].apply(lambda x :float(x))
df['Low'] = df['Low'].apply(lambda x : float(x))

df = round(df,2)
# Remove the date and volume Column in data set
df=df.drop(columns=['Date'])
df=df.drop(columns=['Volume'])
# show the data
print(df.head())

# THERE WE BUILD OUR BUNDLE OF INDICATORS THAT WE ARE GOING TO USE AS INDEPENDENT VARIABLES
# 20 Days Moving average
df["S_20"]=df["Close/Last"].rolling(window=20).mean()
# What rolling() does :
# Calculate the moving average. That is, take the first 20 values, average
# them, then drop the 19 first and add the 20th newly calculated, etc.

# Rolling Correlation
# Here we take 20 days and calculate the correlation between the two features(Close and S_20) in the frame of these 20 days, then we
# iterate one row further for the next 20 days, etc...

df["Corr"]=df["Close/Last"].rolling(window=20).corr(df["S_20"])

# RSI measures speed of change of price movements. RSI value oscillates between 0 and 100. There is a rule
# of thumb that says that values above 70. indicate that the asset is being overbought.

# RSI
df["RSI"]=ta.RSI(np.array(df["Close/Last"]), timeperiod=20)

# The "shift" here indicates that we are subtracting the "open" of the actual day to the "close" of the previous day
df["Open-Close"]=df["Open"]-df["Close/Last"].shift(-1)
df["Open-Open"]=df["Open"]-df["Open"].shift(-1)

df=df.dropna()
print(df.head())

X=df.iloc[:,:9]

# "shift" allows us to subtract every open price with the close price of the previous day

# DEFINING THE DEPENDENT VARIABLE

# Here we take "-1" as parameter to the shift function as we want
# the closing previous day.
# The numpy (np) function "where" creates a new variable called "y" where
# the value is "1" if the close of the today is higher than the close of
# the previous day. In our case we create an array, but an additional column
# could have been created for example by replacing the "y" by "df['Target']"
# for example.

y=np.where(df["Close/Last"].shift(-1) < df["Close/Last"],1,-1)

# SPLITTING THE DATA SET
# We split the data set
# To do this we declare the variable "split" which is going to be an integer
split=int(0.8*len(df))
X_train, X_test, y_train, y_test=X[:split], X[split:], y[:split],  y[split:]

# Instantiate The Logistic Regression in Python

lr = LogisticRegression(solver='lbfgs', max_iter=10000)
lr = lr.fit(X_train, y_train)

#LET’S INTERPRET THE COEFFICIENTS
coef=lr.coef_

# The zip()  function creates the objects and that can be used to
# produce single item at a time. This function can create pandans DataFrames
# by merging two lists.

pd.DataFrame((zip(X.columns,np.transpose(lr.coef_))),columns=['Variable','Coef'])

# CALCULATING CLASS PROBABILITIES
probability = lr.predict_proba(X_test)
print(probability)

#Make predictions using the testing set
predicted = lr.predict(X_test)
print(predicted)

# MODEL EVALUATION
# CONFUSION MATRIX
metrics.confusion_matrix(y_test, predicted)
# Classification Report
print(metrics.classification_report(y_test, predicted))

 # MODEL ACCURACY
lr.score(X_test,y_test)
print(lr.score(X_test,y_test))

# Cross-Validation
cross_val=cross_val_score(LogisticRegression(solver='lbfgs', max_iter=10000), X, y, scoring="accuracy", cv=10)
print(cross_val)
print(cross_val.mean())

# Here I'll just create a data frame for it to be more clear
a = metrics.confusion_matrix(y_test, predicted)
b = pd.Series(['Actual : -1', 'Actual : 1'])
confusion_matrix = pd.DataFrame(a, columns=['Predicted : -1', 'Predicted : 1'])
confusion_matrix = confusion_matrix.set_index([b])
print(confusion_matrix)

# pickle
# Saving stock model to disk
pickle.dump(lr, open('model.pkl','wb'))

# Loading stock model to compare the results
model= pickle.load(open('model.pkl','rb'))
#print(model([[3281.15,3272.87,3297.58,3270.70]]))








