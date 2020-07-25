# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:49:20 2020

@author: Aditya Singh Gaur
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing random selection
import math
N = 10000
d = 10
ads_selected = []
number_of_selections =[0] * d
sum_of_rewards = [0] * d
total_rewards = []
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (number_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i= math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else :
            upper_bound =1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = 1
    number_of_selections[ad] = number_of_selections[ad] + 1
    ads_selected.append(ad)
    reward = dataset.values[n , ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + 1
    total_rewards = total_rewards + reward

#visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of the ads selcted')
plt.xlabel('Ads')
plt.ylabel('Number of times each ads were selected')
plt.show()