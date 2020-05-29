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
    message = [wl.lemmatize(word) for word in message if word not in stopwords.words('english')]
    message = ' '.join(message)
    corpus.append(message)

     
#Creating Bag of Words Model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

#Split the Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size = 0.2,random_state = 0)

#Train the model using NaiveBayes Classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

#Predict the values
y_pred = spam_detect_model.predict(X_test)

#Compare the prediction results with actual results
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test,y_pred)

#Find the Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the model ",accuracy)


