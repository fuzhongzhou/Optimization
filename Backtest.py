#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# In[29]:


pool = pd.read_csv("pool.csv")
pool["Date"] = pd.to_datetime(pool["Date"])
pool = pool.set_index("Date")
pool.head()



def back_test(start, end, commission_fee):
    initial_capital = 10000
    cycle = 12 #transfer frequency, yearly: 12, quaterly: 4
    
    def one_period_trade(t, capital, weight):
        weight = np.array(weight)                                                #target weight
        price = np.array(pool.iloc[t])                                           #current price
        old_position = position_list[-1]                                         #current position
        commission = np.sum(np.abs(capital*weight - old_position*price))*commission_fee  
        capital -= commission                                                   
        position = capital*weight/price
        if t+cycle<end:   #if within back test period 
            next_capital = position.dot(pool.iloc[t+cycle])
        else:             #if not
            next_capital = position.dot(pool.iloc[end])
        position_list.append(position)                                           #record the position over time
        weight_list.append(weight)                                               #record the weight over time
        return next_capital
    
    t = start
    wealth = initial_capital
    
    #records for wealth over time
    cumulative_value = pd.Series({pool.index[t]:wealth})
    weight_list = [np.array([0, 0, 0, 0, 0])]
    position_list = [np.array([0, 0, 0, 0, 0])] 
    
    while(t<end):
        '''
        Hundreds of codes for strategy that generates the new weight
        '''
        
        #WARNING:just for test, check using 5th fund price
        weight = [0, 0, 0, 0, 1]
        
        
        wealth = one_period_trade(t, wealth, weight)
        cumulative_value[pool.index[t]] = wealth
        t += cycle
        
    return cumulative_value


# In[55]:


commission_fee = 0.0002
start = 0            #start date, use the first ordinal number of date index
end = len(pool)-1    #end date, use the last ordinal number date index
result = back_test(start, end, commission_fee)
plt.plot(result)
