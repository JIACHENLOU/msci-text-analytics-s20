#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data_path='/jiachen_lou&shuo_liu_641_final/Supplementary/data'


# In[2]:


from csv import DictReader
import pandas as pd
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


#load train data
f_train_bodies = open(data_path+'/'+'x_train.csv', 'r', encoding='utf-8')
X_train_csv = DictReader(f_train_bodies)
X_train_data = list(X_train_csv)

col_name = ['Headline','Body','Stance']
X_train_headline = [x[col_name[0]] for x in X_train_data]
X_train_body = [x[col_name[1]] for x in X_train_data]
Y_train = [y[col_name[2]] for y in X_train_data]
y_train = Y_train+Y_train


# In[4]:


# Read the text files of fnc data
X_train_df = pd.read_csv(data_path+'/'+'x_train.csv')
X_test_df = pd.read_csv(data_path+'/'+'x_test.csv')


# In[5]:


X_test_df


# In[6]:


# split into training and testing sets

X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_train_df['Body'], X_train_df['Stance'], random_state=10, test_size=0.1)
X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(X_train_df['Headline'], X_train_df['Stance'], random_state=10, test_size=0.1)

X_train = X_train_h + X_train_b
X_val = X_val_h + X_val_b
y_train = y_train_h + y_train_b
y_val = y_val_h + y_val_b
print(len(X_train))
print(len(y_train))
print(len(X_val))
print(len(y_val))


# In[ ]:


#build tfidf train model
X_train_count_vect = CountVectorizer(min_df = 0.2, max_df = 0.7, ngram_range=(1,2))
X_train_count = X_train_count_vect.fit_transform(X_train)
X_train_tfidf_transformer = TfidfTransformer()
X_train_tfidf = X_train_tfidf_transformer.fit_transform(X_train_count)


# In[ ]:


clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[ ]:


def prediction(X_val, clf, count_vect, tfidf_transformer):
    X_val_count = count_vect.transform(X_val)
    X_val_tfidf = tfidf_transformer.transform(X_val_count)
    predict = clf.predict(X_val_tfidf)
    return predict


# In[ ]:


predictions = prediction(X_val, clf, X_train_count_vect, X_train_tfidf_transformer)


# In[ ]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('Accuracy Score',accuracy_score(y_val,predictions))
print('Precision Score',precision_score(y_val,predictions, average='micro'))
print('Recall Score',recall_score(y_val,predictions, average='micro'))
print('F1 Score',f1_score(y_val,predictions, average='micro'))


# In[ ]:


X_test_h = X_test_df['Headline']
X_test_b = X_test_df['Body']
X_test = X_test_h + X_test_b

y_test_h = X_test_df['Stance']
y_test_b = X_test_df['Stance']
y_test = y_test_h + y_test_b


# In[ ]:


predictions = prediction(X_test, clf, X_train_count_vect, X_train_tfidf_transformer)


# In[ ]:


print('Accuracy Score',accuracy_score(y_test,predictions))
print('Precision Score',precision_score(y_test,predictions, average='micro'))
print('Recall Score',recall_score(y_test,predictions, average='micro'))
print('F1 Score',f1_score(y_test,predictions,average='micro'))


# In[ ]:





# In[ ]:




