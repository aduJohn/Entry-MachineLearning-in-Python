# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:54:27 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv('Position_Salaries.csv')
X = data.loc[:,['Level']].values
y = data.loc[:,['Salary']].values

#training the RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

#making a prediction for a random value
y_pred = regressor.predict(np.array(6.5).reshape(-1,1))
print(y_pred)

#plotting the actual RandomForestRegression
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Salary predicted by Level(RandomForestRegression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
