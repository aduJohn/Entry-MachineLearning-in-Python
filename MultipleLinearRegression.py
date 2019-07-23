# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:12:50 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv('50_Startups.csv')
X = data.loc[:,['R&D Spend', 'Administration', 'Marketing Spend', 'State']].values
y = data.loc[:,['Profit']].values

#encoding the data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy trap
X = X[:,1:]

#splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#making the prediction
y_pred = regressor.predict(X_test)

#using the backward elimination
import statsmodels.formula.api as sn
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sn.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#repeat the process
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sn.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sn.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sn.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sn.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()