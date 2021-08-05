import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sklearn

# Technical Indicators
import talib as ta

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# reading CSV file
s3=pd.read_csv(".\HistoricalData_AMZN.csv")

# seting date as a index
s3=pd.read_csv(".\HistoricalData_AMZN.csv",header=0,index_col="Date",parse_dates=True)
# understanding the data
#s1.head()
s3=s3.drop(columns=['Volume'])
print(s3.head())


# # Define Predictor/Independent Variables
# s3["S_10"]=s3["Close/Last"].rolling(window=10).mean()
# s3["Corr"]=s3["Close/Last"].rolling(window=10).corr(s3["S_10"])
# s3["RSI"]=ta.RSI(np.array(s3["Close/Last"]), timeperiod=10)
# s3["Open-Close"]=s3["Open"]-s3["Close/Last"].shift(1)
# s3["Open-Open"]=s3["Open"]-s3["Open"].shift(1)

# removing NA values
s3=s3.dropna()
X=s3.iloc[:,:]


# Define Target/Dependent Variable

y=np.where(s3["Close/Last"].shift(-1) < s3["Close/Last"],1,-1)
# X['y']=X["Close/Last"].apply(lambda x: 1 if x.rolling(window=20).mean() < X["Close/Last"] else -1)
# print(y)
# print(np.unique(y))
#
# sx=pd.DataFrame(y)
# print(sx.value_counts())
# print(s3.describe())


# Split The Dataset
split=int(0.7*len(s3))
X_train, X_test, y_train, y_test=X[:split], X[split:], y[:split],  y[split:]
#print(X_train)

# Instantiate The Logistic Regression in Python
lr=LogisticRegression()
lr=lr.fit(X_train,y_train)

# Examine The Coefficients
pd.DataFrame(zip(X.columns, np.transpose(lr.coef_)))

# Calculate Class Probabilities
probability=lr.predict_proba(X_test)
print(probability)

# Predict Class Labels
predicted=lr.predict(X_test)

# Evaluate The Model
# Confusion Matrix
print(metrics.confusion_matrix(y_test,predicted))

# Classification Report
print(metrics.classification_report(y_test, predicted))

# Model Accuracy
print(lr.score(X_test,y_test))

# Cross-Validation
cross_val=cross_val_score(LogisticRegression(), X, y, scoring="accuracy")
print(cross_val)
print(cross_val.mean())

# pickle
# Saving stock model to disk
pickle.dump(lr, open('model1.pkl','wb'))

# Loading stock model to compare the results
model1= pickle.load(open('model1.pkl','rb'))
#print(model1([[3281.15,3272.87,3297.58,3270.70]]))


















