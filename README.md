# Prediction of Stock Price for Large-cap companies
## Introduction:
This project aims on predicting the future price changes of a stock of a large cap company. This uses the previous prices and financial news related to that particular company.
The data required is taken from Nasdaq.com. 5 years worth stock price data has been gathered for about 20 large cap companies.The datasets used contain data related to stock price, opening, closing values and the highest, lowest price that particular stock reached.
Firstly, Exploratory Data Analysis is performed on the numerical dataset inorder clean up the data, and change all values to desirable datatype and to derive insights from the data. The insights from this numerical data has been plotted as graphs. Then the newsheadlines data is preprocessed, trained and labelled to apply Naive Bayes on the data. The insights derived from these two datasets help predict future stock price changes of that particular stock.

## Exploratory Data Analysis on Stock price dataset
Any dataset that is to be used to perform analysis on, needs to be prepped before we conduct analysis on that.The data available might contain some discrepancies, and might not be consistent, to avoid all these affecting our analysis , Exploratory Data Analysis was performed on the numerical datasets (in csv format).
In our EDA we have,
- Imported the data
- Imported all the required libraries 
- Got to know about the data, Know the head, tail, get a sample from the data. Find the sum of all missing values. As there were no missing values in our data set, we moved on to change data into their desired data types.
- Plotted insights from the EDA.
- we have Open ,Close\Last,high ,low and volumn columns
## Time Series Data:

The stock market data is time series data as in it cahnges by the time .
A time series is a sequence of numerical data points taken at successive equally spaced points in time.In investing, a time series tracks the movement of stock price, over a specified period of time.

## Univariate and Bivariate Analysis
- Univariate analysis has been performed to draw graphs on History of opening,closing ,high and low prices and on History of volume. 
- Bivariate analysis has been done on Closing - Open, Closing - Low price, Closing - High prices.

### Inference from univariate and bivariate analysis: 
- Closing Price is never smaller than Low Price.
- Closing Price is always smaller than High Price.
- Closing Price is sometimes larger or sometimes smaller than the Opening Price.    
 
#### We have performed Resampling(It involves changing the frequency of your time series observations.) and Zooming-in to the data set and Inference: We can say that profit is not predictable for a short amount of time (week,month). There is a chance of gain and loss at the same time if we buy stock for a short period of time.But for long term there is very less chance of getting loss and profit is not that much good. So we can say that it is neutral.

## PreProcessing the data 
Basically it means manipulating the data according to our use so for that we have use some technique:-
- Functions to calculate moving averages
- Used shift function

## Creating the Model and validating it
1)As it is  time series data we can't just split the data into train and test it is not applied for time series data insted of we have to take the data for a specific time using date
so for that we have used iloc() function and specifies the range of the data into the arguments paranthesis  train_x,train_y these two will have all the data basically it is training data
and test_x,test_y these two are testing data .
2)We had created logistic regression model 
- Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.
- Logistic Regression is used when the dependent variable(target) is categorical.
- Logistic regression is easier to implement, interpret, and very efficient to train.
- It not only provides a measure of how appropriate a predictor(coefficient size)is, but also its direction of association (positive or negative).
- It is very fast at classifying unknown records.
### To import the module:-
- from sklearn.linear_model import LogisticRegression 
- After importing the file we have created the model object and then using fit function we are training the data to the model

##### 3)Using text_x we had made a prediction and than compared with the actual value which is test_y for this the accuracy score is 98%

##### 4)Using train_x we had made prediction and than comapred with the actual value which is train_y for this the accuracy score is 99%

5)As per the classification report the F1 for both 0 and 1 is 98% (Basically its harmonic mean
It conveys the balance between the precision and the recall. 2*((precision*recall)/(precision+recall))

<h1 align="center"> News Article Dataset</h1>

## Introduction:-
The News Article dataset has 2015 to 2021 news data with two column dates and headlines.It contains data for different company basically the headlines of the news
In this data we are going to perform sentiment analysis and get the result that whether the stock price will increase or decrease by just headlines of the news. 

## Text preprocessing for sentiment analysis:

To perform sentiment analysis on news headlines, the data must be prepped in advance. The data is loaded and viewed, then, we prepped the data, by converting everything into desired datatypes say dates into date format from string. After this step, we got our dates and respective news headlines as a table. Then Punctuations are removed.
Then contractions were handled, contractions such as i’ll ,i’d etcetera were converted to proper english words i will, i would. 
After this, the text is converted into lower case, so it will be easy for us to proceed further.Then stopwords were removed. Then Lemmatization and Stemming has been performed on the data. Lemmatization removes the grammar tense and transforms each word into its original form. Another way of converting words to its original form is called stemming. While stemming takes the linguistic root of a word, lemmatization is taking a word into its original lemma.
Atlast, the data was tokenized, then a word cloud was made, to display the words with high frequency. 
Summarizing text preprocessing:
Imported , viewed the data.
Converting everything to desired datatypes.
Removal of punctuations.
Contractions were handled.
Lowercase conversion.
Stopwords removal.
Lemmatization and stemming.
Tokenization.
Wordcloud

## Sentiment Analysis:-
Sentiment Analysis or Opinion Mining is a way of finding out the polarity or strength of the opinion (positive or negative) that is expressed in written text, in the case of this project – stock news articles. According to the principle of document level sentiment analysis, each individual document is tagged with its respective polarity. This is generally done by finding the polarities of each individual word/phrase and sentences and combining them to predict the polarity of the whole document. 



### Modelling

* <font size="4.5">We have applied  Machine Learning techniques such as logistic regression,Naïve bayes model to predict the target variable.
 </font>
* <font size="4.5">We have quantified the sentiment  of news headlines with a positive or negative value, called **polarity**.For sentiment analysis  calculation of overall polarity of headline of each date  is required ,for which sentiment function of textblob  is used.  The overall sentiment is inferred/labelled as positive, neutral or negative based on the sign of the polarity score. 
 </font>
* <font size="4.5">The **compound score** is the sum of positive, negative & neutral scores which is then normalized between -1(most extreme negative) and +1 (most extreme positive).The more Compound score closer to **+1**, the **higher** the positivity of the text. 
 </font>
* <font size="4.5">**VADER** is used to quantify how much of ***positive or negative emotion*** the text has and also the intensity of emotion. Vader SentimentIntensityAnalyzer is used in this program to calculate the news headline compound value for a given day.  
 </font>
 

 ### <p> Step 1) Defining the model</p>

<font size="4.5"><span style="color:Black;font-family: Arial">In this step, we generate our model-fitting our dataset in the MultinomialNB.
We will use one of the Naive Bayes (NB) classifier for defining the model. Specifically, we will use 
MultinomialNB classifier. 
We used Naïve Bayes model because it works particularly well with text classification and spam filtering. Advantages of working with NB algorithm are:
    
*	Requires a small amount of training data to learn the parameters
*	Can be trained relatively fast compared to sophisticated models

</span></font>

![image](https://user-images.githubusercontent.com/45910682/128031147-fdddbba5-6401-453f-a655-3cff0fca763e.png)
![image](https://user-images.githubusercontent.com/45910682/128031176-44c3caee-81d9-4a7e-9c79-d8c010967355.png)



### <p> Step 2) Splitting the Dataset</p>

<font size="4.5"><span style="color:Black;font-family: Arial">We do the train/test split before the CountVectorizer to properly simulate the real world where our future data contains words we have not seen before.Split then vectorize is considered the correct way.

</span></font>  

### <p> Step 3) Applying tf vectorizer (count vectorizer)</p>

<font size="4.5"><span style="color:Black;font-family: Arial">If we want to use text in machine learning algorithms, we’ll have to convert them to a numerical representation.   One of the methods is called bag-of-words approach. The bag of words model ignores grammar and order of words. 
Here, we use CountVectorizer (another term of TfVectorizer).it will basically convert these sentences into vectors. That is what bag of words means.
</span></font>

### Now we have the training and testing data. We should start the analysis. 


### <p> Step 4) Applying Naive Bayes</p>

<font size="4.5"><span style="color:Black;font-family: Arial">Training Naive Bayes classifier with train data.Since we are using sklearn’s modules and classes we just need to import the precompiled classes.we generate our model-fitting our dataset in the MultinomialNB.Naive Bayes using bag of words (unigrams of words) and character level n-grams. N-gram  is  a  tokenization  process  to  separate  words  based on the type of token used. In this study mainly bigram was used. Bigram  is  a  n-word  solution  in  a  review  sentence  with  n  =  2. ngram_range=(2,2)
</span></font>

### <p> Step 5) Evaluation</p>

<font size="4.5"><span style="color:Black;font-family: Arial">The performance evaluation of the Naive Bayes algorithm in this study was carried out by calculating the value of Precision, Recall, Accuracy and Error Rate by using confusion matrix. Here we are applying the classification report, confusion matrix, and accuracy score to check the accuracy of our model.Cross-validation is a statistical method used to estimate the skill of machine learning models.
</span></font>

### <p> Step 5)  Improving the model and handling unbalanced data</p>

<font size="4.5"><span style="color:Black;font-family: Arial">As is known to us, Naive Bayes algorithm is a simple and efficient categorization algorithm. However, the assumption of conditional independence in this algorithm does not conform to objective reality which affects its categorization performance to some extent. In order to improve the categorization performance of Naive Bayes algorithm in text categorization, a Naive Bayes text categorization algorithm based on TF-IDF attribute weighting  is used. Let’s use TF-IDF, which takes in account the product of term frequency and inverse document frequency. TF-IDF shows the rarity of a word in the corpus. If a word is rare then probably its a signature word for a particular sentiment/information.

</span></font> 
 
#### After calculation of accuracy and generation of classification report for sentiment analysis, from test data from  Naive Bayes classifier model ; We got the accuracy of 91.54% .




## Deployment of the model:
 ### Fronted 
 Technology used Flask:-
 - Flask is a web framework. This means flask provides you with tools, libraries and technologies that allow you to build a web application. This web application can be some web pages, a blog, a wiki or go as big as a web-based calendar application or a commercial website
 - HTML/CSS
### Hosting 
Technology used Heroku:-
- Heroku is a container-based cloud Platform as a Service (PaaS). it use for to deploy, manage, and scale modern apps. 
Our platform is elegant, flexible, and easy to use, offering developers the simplest path to getting their apps to market.

## Deployed model on Heroku:-
![WhatsApp Image 2021-08-05 at 20 09 57](https://user-images.githubusercontent.com/45910682/128371907-52c7a8a8-1c67-43f2-a4bf-4bf0a08dd11f.jpeg)

After clicking on News Prediction button....

![WhatsApp Image 2021-08-05 at 20 10 23](https://user-images.githubusercontent.com/45910682/128371893-9dd2b968-6a2b-4a20-8651-d75396a97384.jpeg)



    








