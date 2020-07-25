# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 22:01:57 2020

@author: Aditya Singh Gaur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#reading the dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv')
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    

#training apriori into the dataset
from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#visualising the results
resuts = list(rules)

