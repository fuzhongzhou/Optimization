import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BlackLitterman as BL
import RiskParity as RP
from BlackLitterman import BlackLitterman, MeanVariance, MeanVarianceConstraint
from RiskParity import risk_parity_weight, standardize, RiskContribution
from Backtest import back_test

# read data
pool_raw = pd.read_csv("pool.csv", encoding='utf-8', index_col=0)[:-1]
pool = pool_raw.copy()
pool = pool.fillna(pool.mean())
for i in pool.columns:
    pool[i] = standardize(pool[i])
pool.head()
ret = (pool_raw / pool_raw.shift(1) - 1)
n = pool_raw.shape[1]

#####################
win = 18
winv = 6
cycle = 4

date = list(ret.index)
date_idx = list(range(len(ret.index)))
trade_date = date[win::cycle]
trade_idx = date_idx[win::cycle]

weight = pd.DataFrame(columns=ret.columns, index=trade_date)
weight_rp = weight.copy(deep=True)
weight_mv = weight.copy(deep=True)

output = 0


for d in trade_idx:
    print(d)
    cov = pool.iloc[d-win+1:d+1, :].cov()
    Sig = ret.iloc[d-win+1:d+1, :].cov()
    # Sig = cov.copy()
    ER = ret.iloc[d-win+1:d+1, :].mean()  # / ret.std()

    ################ Discretionary Parameters
    # risk parity params
    target = [0.7, 0.198, 0.1, 0.002]  # target risk contribution of equity, bond, alternative, liquidity
    equity = 0.8  # equity proportion limit
    liquidity_interval = (0.05, 0.1)  # liquidity proportion interval

    # black litterman params
    rf = 0.0 / 12
    tau = 0.01  # 1 / n, n is # of observation

    P = []
    for i in range(n):
        tmp = [0] * n
        tmp[i] = 1
        P.append(tmp)
    P = np.array(P)

    Q = []
    for i in range(n):
        Q.append([ret.iloc[d-winv+1:d+1, i].mean()])

    ################ Time for models

    w_riskparity = risk_parity_weight(cov, target, equity, liquidity_interval)
    risk_contribution = RiskContribution(w_riskparity, cov)

    if output:
        print("risk parity weight")
        print(w_riskparity)
        print("risk parity risk contribution")
        print(risk_contribution)


    w_mkt, lam_mkt = MeanVariance(ER, Sig, rf)
    w_BL, implied_confidence = BlackLitterman(w_riskparity, ER, Sig, lam_mkt, rf, tau, P, Q)
    risk_contribution_bl = RiskContribution(w_BL, cov)

    if output:
        print("black litterman weight")
        print(w_BL)
        print("black litterman implied confidence")
        print(implied_confidence)
        print("black litterman risk contribution")
        print(risk_contribution_bl)

    weight.loc[date[d]] = w_BL.reshape((1, -1))
    weight_rp.loc[date[d]] = np.array(w_riskparity)
    weight_mv.loc[date[d]] = np.array(w_mkt).reshape((1, -1))

weight.to_csv('weight.csv')

weight_eq = pd.DataFrame(columns=ret.columns, index=trade_date)
weight_eq = weight_eq.fillna(1 / 5)
weight_eq.to_csv("eq_weight.csv")



commission_fee = 0.0002
start = trade_idx[0]            #start date, use the first ordinal number of date index
end = trade_idx[-1]  #end date, use the last ordinal number date index
result_eq = back_test(weight_eq, start, end, commission_fee)
result_rp = back_test(weight_rp, start, end, commission_fee)
result_mv = back_test(weight_mv, start, end, commission_fee)
result_bl = back_test(weight, start, end, commission_fee)

plt.plot(result_eq)
plt.plot(result_rp)
plt.plot(result_mv)
plt.plot(result_bl)
#plt.yticks(np.array(result_bl.index))
plt.legend(['equally weighted', 'risk parity', 'mean variance', 'black litterman'])
plt.show()