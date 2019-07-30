# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:42:41 2019

@author: Alexandru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions = []
for i in range(0,max(data.index)+1):
    transactions.append([str(data.values[i,j]) for j in range(0,len(data.values[i,:]))])

from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,max_length=2)

results = list(rules)