#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize


def risk_parity_weight(cov_mat, risk_prop, equity_prop):    
    #risk parity weight
    
    #cov_mat: covariance matrix
    #risk_prop: target risk contribution proportion, a list
    #equity_prop: the equity proportion limit
    
    #WARNING: the following part needs to be customized manually
    #1. delta_risk in objfun(x), need to be adjusted with respect to the distribution for type of funds and corresbonding target
    #2. cons, need to be adjusted with respect to the distribution for equity funds
    
    sigma = np.matrix(cov_mat.values)

    def objfun(x):
        tmp = (sigma * np.matrix(x).T).A1
        risk = x * tmp
        var = sum(risk)
        delta_risk = ((sum(risk[0:3]) - risk_prop[0]*var)**2 + 
                      (sum(risk[3:5]) - risk_prop[1]*var)**2 +
                      (sum(risk[5:9]) - risk_prop[2]*var)**2)
        return delta_risk

    x0 = np.ones(sigma.shape[0]) / sigma.shape[0]  
    bnds = tuple((0,None) for x in x0)
    cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: equity_prop - sum(x[0:3])/x.sum()})
    options={'disp':False, 'maxiter':1000, 'ftol':1e-20}

    # optimization
    res = minimize(objfun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    wts = pd.Series(index=cov_mat.index, data=res['x'])
    wts = wts / wts.sum() * 1.0

    # risk contribution
    var = (wts.T).dot(cov_mat).dot(wts)
    risk_contribution = wts * (cov_mat.dot(wts)) / var
    risk_contribution = [sum(risk_contribution[0:3]), sum(risk_contribution[3:5]), sum(risk_contribution[5:9])]

    return wts, risk_contribution



def standardize(tmp):
    #standardization, z-score approach
    #tmp: input, a pd.series from a pandas 
    if isinstance(tmp, pd.Series):
        mu = tmp.mean()
        sigma = tmp.std()
        tmp = (tmp - mu)/sigma
    return tmp   



if __name__ == "__main__":
    pool=pd.read_csv("pool.csv", encoding='utf-8')[:-1]
    pool = pool.set_index("Date")
    pool = pool.fillna(pool.mean())
    for i in pool.columns:
        pool[i] = standardize(pool[i])
    pool.head()


    cov = pool.cov()
    target = [0.7, 0.2, 0.1] # target risk contribution of equity, bond, alternative
    equity = 0.8 # equity proportion limit
    wts = risk_parity_weight(cov, target, equity)
    wts


    var = (wts.T).dot(cov).dot(wts)
    risk_contribution = wts*(cov.dot(wts))/var
    risk_contribution = [sum(risk_contribution[0:3]), sum(risk_contribution[3:5]), sum(risk_contribution[5:9])]
    risk_contribution

