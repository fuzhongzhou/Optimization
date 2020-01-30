#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import BlackLitterman as BL
import RiskParity as RP
from BlackLitterman import BlackLitterman, MeanVariance, MeanVarianceConstraint
from RiskParity import risk_parity_weight, standardize, RiskContribution


# In[29]:






def back_test(df, start, end, commission_fee):
    pool = pd.read_csv("pool.csv")
    # pool["Date"] = pd.to_datetime(pool["Date"])
    pool = pool.set_index("Date")
    pool.head()
    date = np.array(pool.index)
    date_idx = np.array(range(len(pool.index)))


    initial_capital = 10000
    cycle = 4 #transfer frequency, yearly: 12, quaterly: 4
    
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
    
    while t < end:
        '''
        Hundreds of codes for strategy that generates the new weight
        '''
        
        #WARNING:just for test, check using 5th fund price
        weight = df.loc[date[t]]
        
        wealth = one_period_trade(t, wealth, weight)
        cumulative_value[pool.index[t]] = wealth
        t += cycle
        
    return cumulative_value


# In[55]:


if __name__ == '__main__':
    pool = pd.read_csv("pool.csv")
    # pool["Date"] = pd.to_datetime(pool["Date"])
    pool = pool.set_index("Date")
    pool.head()

    weight = pd.read_csv('weight.csv', index_col=0)
    eq_weight = pd.read_csv('eq_weight.csv', index_col=0)
    date = np.array(pool.index)
    date_idx = np.array(range(len(pool.index)))

    trade_date = np.array(weight.index)
    trade_idx = np.array([np.argwhere(date == i)[0, 0] for i in trade_date])


    commission_fee = 0.0002
    start = trade_idx[0]            #start date, use the first ordinal number of date index
    end = trade_idx[-1]  #end date, use the last ordinal number date index
    result_eq = back_test(eq_weight, start, end, commission_fee)
    result = back_test(weight, start, end, commission_fee)

    plt.plot(result_eq)
    plt.plot(result)
    plt.legend(['eq', 're'])
    plt.show()

    pass
