# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:24:33 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')
X = data.loc[:,['EstimatedSalary','Age']].values
y = data.loc[:,['Purchased']].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)