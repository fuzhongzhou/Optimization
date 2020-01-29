import numpy as np
import pandas as pd
import BlackLitterman as BL
import RiskParity as RP
from BlackLitterman import BlackLitterman, MeanVariance, MeanVarianceConstraint
from RiskParity import risk_parity_weight, standardize, RiskContribution

# read data
pool_raw = pd.read_csv("pool.csv", encoding='utf-8', index_col=0)[:-1]
pool = pool_raw.copy()
pool = pool.fillna(pool.mean())
for i in pool.columns:
    pool[i] = standardize(pool[i])
pool.head()
ret = (pool_raw / pool_raw.shift(1) - 1)[1:]

################ Computed parameters
cov = pool[-18:].cov()
Sig = ret[-18:].cov()
# Sig = cov.copy()
ER = ret[-18:].mean()# / ret.std()

################ Discretionary Parameters
# risk parity params
target = [0.7, 0.2, 0.1]  # target risk contribution of equity, bond, alternative
equity = 0.8  # equity proportion limit

# black litterman params
rf = 0.03 / 12
tau = 0.01  # 1 / n, n is # of observation

P = np.array([[0, 0, 1, -1, 0, 0, 0, 0, 0],
              [2, -1, -1, 0, 0, 0, 0, 0, 0]])
Q = np.array([[0.01],
              [0.02]])
confidence = np.array([[0.5],
                       [0.4]])


################ Time for models

w_riskparity = risk_parity_weight(cov, target, equity)
risk_contribution = RiskContribution(w_riskparity, cov)

print("risk parity weight")
print(w_riskparity)
print("risk parity risk contribution")
print(risk_contribution)

pass

w_mkt, lam_mkt = MeanVariance(ER, Sig, rf)
w_BL, implied_confidence = BlackLitterman(w_riskparity, Sig, lam_mkt, rf, tau, P, Q)
risk_contribution_bl = RiskContribution(w_BL, cov)

print("black litterman weight")
print(w_BL)
print("black litterman implied confidence")
print(implied_confidence)
print("black litterman risk contribution")
print(risk_contribution_bl)

pass

#print(MeanVarianceConstraint(Sig, ER, rf, lam_mkt))
