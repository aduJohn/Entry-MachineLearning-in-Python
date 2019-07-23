# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:58:30 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data
data = pd.read_csv("Salary_Data.csv")
X = data.loc[:,['YearsExperience']].values
y= data.loc[:,['Salary']].values

#splittin the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#creating the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the data
y_pred = regressor.predict(X_test)

#plotting the data for the training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#plotting the data for the testing set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()