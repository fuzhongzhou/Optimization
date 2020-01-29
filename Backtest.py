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
pool = pool[["Wells Fargo C&B Large Cap Value A", "Metropolitan West Total Return Bd I",
            "Vanguard Real Estate ETF", "iShares Gold Trust"]]
pool.head()



def back_test(start, end):
    initial_capital = 10000
    cycle = 12 #transfer frequency, yearly: 12, quaterly: 4
    
    def one_period_trade(t, capital, weight):
        price = pool.iloc[t]
        position = capital * weight / np.array(pool.iloc[t])
        if t + cycle < end:   #if within back test period
            capital = position.dot(pool.iloc[t + cycle])
        else:             #if not
            capital = position.dot(pool.iloc[end])
        return capital
    
    #WARNING:just for test, check using 4th fund price
    weight = np.array([0, 0, 0, 1])
    
    t = start
    wealth = initial_capital
    
    #records for wealth over time
    cumulative_value = pd.Series({pool.index[t]:wealth})

    # at time t, change position, compute the return between (t, t + cycle)
    while t < end:
        '''
        Hundreds of codes for strategy which generate the new weight
        
        先跑一遍，把各个时刻的weight都记下来，然后back test直接按着记好的信号搞就可以
        '''
        wealth = one_period_trade(t, wealth, weight)
        cumulative_value[pool.index[t + cycle]] = wealth
        t += cycle
        
    return cumulative_value


# In[55]:


result = back_test(0, len(pool)-1)
plt.plot(result)

