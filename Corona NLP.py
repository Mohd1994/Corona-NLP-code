# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:09:12 2022

@author: user
"""

import pandas as pd
import numpy as np
import nltk
import re

data1=pd.read_csv("Corona_NLP_train.csv",encoding='ISO-8859-1',usecols=['OriginalTweet','Sentiment'])
data2=pd.read_csv("Corona_NLP_test.csv",encoding='ISO-8859-1',usecols=['OriginalTweet','Sentiment'])
df=pd.concat([data1,data2])

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

print(df['OriginalTweet'].isna().sum())

corpus = []
for i in range(0, 44955):
    review = re.sub('[^a-zA-Z]', ' ', df['OriginalTweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer

#cv = CountVectorizer(max_features=1000) # Best accuracy = 98.3%, SVC linear.

#cv = CountVectorizer(max_features=1500) # Best accuracy = 98.6%, SVC linear.

#cv = CountVectorizer(max_features=2000) # Best accuracy = 98.8%, SVR linear. 

cv = CountVectorizer(max_features=2500)  # Best accuracy = 98.85% SVR linear. 

#cv = CountVectorizer(max_features=3000) # Best accuracy = 98.7%, SVR linear. 

#cv = CountVectorizer(max_features=3500) # Best accuracy = 98.7%, SVR linear. 

#cv = CountVectorizer(max_features=4000) # Best accuracy = 98.85% SVR linear. 

#cv = CountVectorizer(max_features=4500) # Best accuracy = 98.8%, SVR linear.

#cv = CountVectorizer(max_features=5000) # Best accuracy = 98.6%, SVR linear. 

X=cv.fit_transform(clean).toarray()

y = df.iloc[:,-1].values


"""
After defining our bag of data, we can start training our model. We can choose
any classification method we want.
"""


# ------------------------------------------------------------------------- #
# ---------------------------- DECISION TREE ------------------------------ #
# ------------------------------------------------------------------------- #


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #1 ######################")
print("################### DECISION TREE ######################\n")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)

from sklearn.metrics import accuracy_score

print(f"\Accuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# ------------------------- LOGISTIC REGRESSION --------------------------- #
# ------------------------------------------------------------------------- #



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #2 ######################")
print("################### LOGISTIC REGRESSION ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# --------------------------------- KNN ----------------------------------- #
# ------------------------------------------------------------------------- #



from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #3 ######################")
print("################### KNN ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# ----------------------------- NAIVE BAYES ------------------------------- #
# ------------------------------------------------------------------------- #



from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #4 ######################")
print("################### NAIVE BAYES ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# ---------------------------- RANDOM FOREST ------------------------------ #
# ------------------------------------------------------------------------- #



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=5)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #5 ######################")
print("################### RANDOM FOREST ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# ------------------------------- SVC RBF --------------------------------- #
# ------------------------------------------------------------------------- #



from sklearn.svm import SVC

classifier = SVC()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #6 ######################")
print("################### SVC RBF ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# ----------------------------- SVC LINEAR -------------------------------- #
# ------------------------------------------------------------------------- #



classifier = SVC(kernel='linear')

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #7 ######################")
print("################### SVC LINEAR ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# ------------------------------- SVC POLY -------------------------------- #
# ------------------------------------------------------------------------- #



classifier = SVC(kernel='poly')

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #8 ######################")
print("################### SVC POLY ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")



# ------------------------------------------------------------------------- #
# ----------------------------- SVC SIGMOID ------------------------------- #
# ------------------------------------------------------------------------- #



classifier = SVC(kernel='sigmoid')

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print("\n################### TEST #9 ######################")
print("################### SVC SIGMOID ######################\n")

print(cm)

print(f"\nAccuracy score: {accuracy_score(y_test,y_pred)}")










