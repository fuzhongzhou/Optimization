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
n = pool_raw.shape[1]

#####################
win = 18
cycle = 4
date = list(range(len(ret.index)))[win:]
trade_date = date[::cycle]

################ Computed parameters

winv = 6
cov = pool[-win:].cov()
Sig = ret[-win:].cov()
# Sig = cov.copy()
ER = ret[-win:].mean()# / ret.std()
weight = pd.DataFrame(columns=ret.columns, index=trade_date)

for d in trade_date:
    ################ Discretionary Parameters
    # risk parity params
    target = [0.7, 0.198, 0.1, 0.002]  # target risk contribution of equity, bond, alternative, liquidity
    equity = 0.8  # equity proportion limit
    liquidity_interval = (0.05, 0.1)  # liquidity proportion interval

    # black litterman params
    rf = 0.01 / 12
    tau = 1 / win  # 1 / n, n is # of observation

    P = np.array([[0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
                  [2, -1, -1, 0, 0, 0, 0, 0, 0, 0]])
    Q = np.array([[0.01],
                  [0.02]])

    P = []
    for i in range(n):
        tmp = [0] * n
        tmp[i] = 1
        P.append(tmp)
    P = np.array(P)

    Q = []
    for i in range(n):
        Q.append([ret.iloc[d-winv:d, i].sum()])

    ################ Time for models

    w_riskparity = risk_parity_weight(cov, target, equity, liquidity_interval)
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

    weight.loc[d] = w_BL

    #print(MeanVarianceConstraint(Sig, ER, rf, lam_mkt))
