# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:26:08 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv('Position_Salaries.csv')
X = data.loc[:,['Level']].values
y=data.loc[:,['Salary']].values

#scalling the data
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
y_sc = StandardScaler()
X = X_sc.fit_transform(X)
y = y_sc.fit_transform(y)

#creating the SVR model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)
#predicting the salary for the level of 6.5
y_pred = y_sc.inverse_transform(regressor.predict(X_sc.transform(np.array(6.5).reshape(-1,1))))
print(y_pred)

#plotting the SVR model
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X_sc.fit_transform(X)),color='blue')
plt.title('Salary based on level(Using SVR)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
