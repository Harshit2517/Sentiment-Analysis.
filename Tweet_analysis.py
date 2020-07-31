# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:42:24 2020

@author: harsh
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps= PorterStemmer()

dataset= pd.read_csv('train.csv')

uncleaned= dataset['tweet'].values
cleaned=[]
uncleaned.size

for i in range(uncleaned.size):
    text= re.sub('[^a-zA-Z]', ' ', uncleaned[i])
    text= text.lower()
    text= text.split()
                       
    text= [ps.stem(word)for word in text if not word in set(stopwords.words('english'))]
    text= ' '.join(text)
                            
    cleaned.append(text)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 1000)

X= cv.fit_transform(cleaned)
X= X.toarray()

y= dataset['label'].values

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test= train_test_split(X, y)

from sklearn.linear_model import LinearRegression, LogisticRegression
lin_r= LinearRegression()
log_r= LogisticRegression()

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()

from sklearn.svm import SVC
sv= SVC()

from sklearn.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier()

log_r.fit(X_train, y_train)
nb.fit(X_train, y_train)
sv.fit(X_train, y_train)
dtf.fit(X_train, y_train)

log_r.score(X_test, y_test)
nb.score(X_test, y_test)
sv.score(X_test, y_test)
dtf.score(X_test, y_test)                      

from sklearn.metrices import confusion_matrix

cm_log= confusion_matrix(y_test, y_log)
cm_nb= confusion_matrix(y_test, y_nb)
cm_sv= confusion_matrix(y_test, y_sv)
cm_dtf= confusion_matrix(y_test, y_dtf)