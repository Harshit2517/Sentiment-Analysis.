# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:06:10 2020

@author: harsh
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('drugLibTrain_raw.tsv', sep= '\t')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps= PorterStemmer()

unc_benifits= dataset['benefitsReview'].values
cleaned_ben=[]

                        for i in range(unc_benifits.size):
                            text= re.sub('[^a-zA-Z]', ' ', unc_benifits[i])
                            text= text.lower()
                            text= text.split()
                            
                            text= [ps.stem(word)for word in text if word not in set(stopwords.words('english'))]
                            text= ' '.join(text)
                            
                            cleaned_ben.append(text)


unc_side= dataset['sideEffectsReview'].values
cleaned_eff=[]
unc_side= pd.DataFrame(unc_side)
from sklearn.impute import SimpleImputer
sm= SimpleImputer(missing_values= np.nan, strategy='most_frequent')

unc_side= sm.fit_transform(unc_side)
unc_side= unc_side[:,0]

                        for i in range(unc_side.size):
                            text= re.sub('[^a-zA-Z]', ' ', unc_side[i])
                            text= text.lower()
                            text= text.split()
                            
                            text= [ps.stem(word)for word in text if word not in set(stopwords.words('english'))]
                            text= ' '.join(text)
                            
                            cleaned_eff.append(text)
                            
unc_comment= dataset['commentsReview'].values
cleaned_comments=[]
unc_comment= pd.DataFrame(unc_comment)
unc_comment= sm.fit_transform(unc_comment)
unc_comment= unc_comment[:,0]

                         for i in range(unc_comment.size):
                            text= re.sub('[^a-zA-Z]', ' ', unc_comment[i])
                            text= text.lower()
                            text= text.split()
                            
                            text= [ps.stem(word)for word in text if word not in set(stopwords.words('english'))]
                            text= ' '.join(text)
                            
                            cleaned_comments.append(text)
                            

y= dataset.iloc[:,2].values


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 500)
X= cv.fit_transform(cleaned_ben)
X= X.toarray()

X2= cv.fit_transform(cleaned_comments)
X2= X2.toarray()

X3= cv.fit_transform(cleaned_eff)
X3= X3.toarray()

X= np.concatenate((X, X2, X3), axis=0)
X= X.reshape(3107, 1500)

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
                           
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test= train_test_split(X, y)


from sklearn.tree import DecisionTreeClassifier
dtf= DecisionTreeClassifier()


nb.fit(X_train, y_train)
nb.score(X_test, y_test)

