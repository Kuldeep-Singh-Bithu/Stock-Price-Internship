import os
import glob
import matplotlib
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
os.chdir("headlines/headlinesOfAllYears")
extension = "CSV"
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_headline=pd.concat([pd.read_csv(f) for f in all_filenames])
#export to csv
combined_headline.to_csv("combined_headline.csv",index=False)
# Removing punctuations
data=combined_headline["Headlines"]
data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)
# Converting headlines to lower case
data=data.str.lower()

print(data.head())





