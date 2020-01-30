#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize


def risk_parity_weight(cov_mat, risk_prop, equity_prop, liquidity):    
    #risk parity weight
    
    #cov_mat: covariance matrix
    #risk_prop: a list, target risk contribution proportion
    #equity_prop: a float, the equity proportion limit
    #liquidity: a tuple, the interval of liquidity
    
    #WARNING: the following part needs to be customized manually
    #1. delta_risk in objfun(x), need to be adjusted with respect to the distribution for type of funds and corresbonding target
    #2. cons, need to be adjusted with respect to the distribution for equity funds, alternative and liquidity
    sigma = np.matrix(cov_mat.values)

    def objfun(x):
        tmp = (sigma * np.matrix(x).T).A1
        risk = x * tmp
        var = sum(risk)
        delta_risk = ((risk[0] - risk_prop[0]*var)**2 +           #equity
                      (risk[1] - risk_prop[1]*var)**2 +           #bond
                      (sum(risk[2:4]) - risk_prop[2]*var)**2 +    #alternative
                      (risk[4] - risk_prop[3]*var)**2)            #liquidity 
        return delta_risk

    x0 = np.ones(sigma.shape[0]) / sigma.shape[0]  
    bnds = tuple((0,None) for x in x0)
    cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1},                               #total limit = 1
            {'type': 'ineq', 'fun': lambda x: equity_prop - (x[0]+x[2:4])/x.sum()},   #equity <= equity_prop
            {'type': 'ineq', 'fun': lambda x: liquidity[1] - x[4]/x.sum()},           #liquidity <=  
            {'type': 'ineq', 'fun': lambda x: x[4]/x.sum() - liquidity[0]})           #liquidity >= 
    options={'disp':False, 'maxiter':1000, 'ftol':1e-20}

    # Optimization
    res = minimize(objfun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    wts = pd.Series(index=cov_mat.index, data=res['x'])
    return wts / wts.sum() * 1.0



def standardize(tmp):
    #standardization, z-score approach
    #tmp: input, a pd.series from a pandas 
    if isinstance(tmp, pd.Series):
        mu = tmp.mean()
        sigma = tmp.std()
        tmp = (tmp - mu)/sigma
    return tmp   


def RiskContribution(wts, cov_mat):
    var = (wts.T).dot(cov_mat).dot(wts)
    risk_contribution = np.array(wts * (cov_mat.dot(wts)) / var)
    risk_contribution = [risk_contribution[0], risk_contribution[1], sum(risk_contribution[2:4]), risk_contribution[4]]
    return risk_contribution


if __name__ == "__main__":
    pool=pd.read_csv("pool.csv", encoding='utf-8')[:-1]
    pool = pool.set_index("Date")
    pool = pool.fillna(pool.mean())
    for i in pool.columns:
        pool[i] = pool[i]/pool[i].iloc[0]
    pool.head()

    cov = pool.cov()
    target = [0.7, 0.198, 0.1, 0.002] # target risk contribution of equity, bond, alternative, liquidity
    equity = 0.8                      # equity proportion limit
    liquidity_interval = (0.05, 0.1)  # liquidity proportion interval
    wts = risk_parity_weight(cov, target, equity, liquidity_interval)
    print(wts)
    
    var = (wts.T).dot(cov).dot(wts)
    risk_contribution = wts*(cov.dot(wts))/var
    risk_contribution = [risk_contribution[0], risk_contribution[1], sum(risk_contribution[2:4]), risk_contribution[4]]
    print(risk_contribution)

