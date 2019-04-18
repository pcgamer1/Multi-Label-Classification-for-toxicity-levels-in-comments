# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:57:27 2019

@author: Sarthak
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

#Fetching Data
train = pd.read_csv('C:\\Users\\Sarthak\\.spyder-py3\\train.csv')
test = pd.read_csv('C:\\Users\\Sarthak\\.spyder-py3\\test.csv')

#Defining the none column
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
train.describe()

#Replacing NAN values
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

#Cleaning the comments
train.comment_text.replace("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])","",regex=True,inplace=True)
train.comment_text.replace("(<br\s*/><br\s*/>)|(\-)|(\/)"," ",regex=True,inplace=True)
test.comment_text.replace("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])","",regex=True,inplace=True)
test.comment_text.replace("(<br\s*/><br\s*/>)|(\-)|(\/)"," ",regex=True,inplace=True)

#Creating One-Hot vectorized Train and Test sets using tfid vectorizer

tfv = TfidfVectorizer(min_df=3, max_df=0.9, max_features=None, strip_accents='unicode',\
               analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), use_idf=1,\
               smooth_idf=1, sublinear_tf=1, stop_words='english')
print("Processing train set")
tfv.fit(train['comment_text'])
xtrn = tfv.transform(train['comment_text'])
print("Processing test set")
xtst = tfv.transform(test['comment_text'])

#Logistic Regression Model
def get_mdl(y):
    y = y.values
    m = LogisticRegression()
    return m.fit(xtrn,y)

#Initializing predicted values to 0
preds = np.zeros((len(test), len(label_cols)))

#Performing label wise cassification and printing the cross validation score for each label
for i, j in enumerate(label_cols):
    print('fit', j)
    m = get_mdl(train[j])
    preds[:,i] = m.predict_proba(xtst)[:,1]
    print(cross_val_score(m, xtrn, train[j], cv=3, scoring='f1'))
