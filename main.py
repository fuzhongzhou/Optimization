import numpy as np
import pandas as pd
import BlackLitterman as BL
import RiskParity as RP
from BlackLitterman import BlackLitterman, MeanVariance, MeanVarianceConstraint
from RiskParity import risk_parity_weight, standardize

# read data
pool = pd.read_csv("pool.csv", encoding='utf-8')[:-1]
pool = pool.set_index("Date")
pool = pool.fillna(pool.mean())
for i in pool.columns:
    pool[i] = standardize(pool[i])
pool.head()


cov = pool.cov()
target = [0.7, 0.2, 0.1]  # target risk contribution of equity, bond, alternative
equity = 0.8  # equity proportion limit
wts, risk_contribution = risk_parity_weight(cov, target, equity)
print(wts, risk_contribution)







######################### testing inputs, will be replaced
Sig = cov
ER = pool.mean(axis=0)
rf = 0.00

# Black Litterman Parameters
tau = 0.05  # 1 / n, n is # of observation

# View parameters
P = np.array([[0, 0, 1, -1, 0, 0, 0, 0, 0],
              [-2, 1, 1, 0, 0, 0, 0, 0, 0]])
Q = np.array([[0.001],
              [0.005]])
confidence = np.array([[0.5],
                       [0.4]])

#############################################


w_mkt, lam_mkt = MeanVariance(ER, Sig, rf)
print(BlackLitterman(w_mkt, Sig, lam_mkt, rf, tau, P, Q))
print(w_mkt)

print(MeanVarianceConstraint(Sig, ER, rf, lam_mkt))
