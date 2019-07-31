# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:52:38 2019

@author: Alexandru
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13].values
y = data.iloc[:,13].values

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_1 = LabelEncoder()
label_encoder_2 = LabelEncoder()
X[:,1] = label_encoder_1.fit_transform(X[:,1])
X[:,2] = label_encoder_2.fit_transform(X[:,2])
one_hot_encoder = OneHotEncoder(categorical_features = [1])
X = one_hot_encoder.fit_transform(X).toarray()

#escaping the dummy trap
X = X[:,1:]

#splitting the data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#scalling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing the keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#Innitialing the ANN
classifier = Sequential()

#Adding the first hidden layer and the input layer
classifier.add(Dense(output_dim = 6,init='uniform',activation='relu',input_dim=11))

#adding another hidden layer
classifier.add(Dense(output_dim = 6,init='uniform',activation='relu'))

#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling the ann
classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#make the final prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#seeing the result with the confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#calculating the accuracy
accuracy = ((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))
print(accuracy)