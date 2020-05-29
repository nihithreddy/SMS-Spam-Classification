# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:53:43 2020

@author: Nihith
"""

#import the libraries
import pandas as pd
#Create a Dataframe Object
messages = pd.read_csv("smsspamcollection/SMSSpamCollection",sep="\t",names=["label","message"])

#Data Cleaning and Data Preprocessing
import re
import nltk
#Download stopwords 
nltk.download('stopwords')


#import the libraries for doing stemming,lemmatization 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#Creating an Object of PorterStemmer
ps = PorterStemmer()
#Create an Object of WordNetLemmatizer
wl = WordNetLemmatizer()
#Create a corpus for storing the final messages after data preprocessing
corpus = []
for i in range(len(messages)):
    #Remove all the numbers and punctuations which are not neccessary
    message = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if word not in stopwords.words('english')]
    message = ' '.join(message)
    corpus.append(message)

     
    