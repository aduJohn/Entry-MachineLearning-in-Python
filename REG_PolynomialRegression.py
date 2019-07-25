# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:12:00 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv('Position_Salaries.csv')
X = data.loc[:,['Level']].values
y = data.loc[:,['Salary']].values

#creating the linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#creating the polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#plotting the result for the linear regression model
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Salary based on level(LinearRegression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#plotting the result for the polynomial regression model
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Salary based on level(PolynomialRegression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Predicting the 6.5 with Linear Regression
var =np.array(6.5)
var =var.reshape(-1,1)
lin_reg.predict(var)

# Predicting the 6.5 with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(var))
