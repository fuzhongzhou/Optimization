import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BlackLitterman as BL
import RiskParity as RP
from numpy import exp
from BlackLitterman import BlackLitterman, MeanVariance, MeanVarianceConstraint
from RiskParity import risk_parity_weight, standardize, RiskContribution
from Backtest import back_test
plt.style.use('ggplot')
import matplotlib.dates as mdates
import datetime

from Backtest import back_test, sharpe_ratio, maximum_drawdown


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
equity = 0.8
liquidity_top = 0.1

for d in trade_idx:
    print(d)
    cov = pool.iloc[d-win+1:d+1, :].cov()
    Sig = ret.iloc[d-win+1:d+1, :].cov()
    # Sig = cov.copy()
    ER = ret.iloc[d-win+1:d+1, :].mean()  # / ret.std()

    ################ Discretionary Parameters
    # risk parity params
    equity -= 0.0125*cycle/12  # equity proportion limit
    target_equity = (exp(equity)/2 - 0.2)*0.9
    target_alternative = (exp(equity)/2 - 0.2)*0.1
    target_bond = (1-(exp(equity)/2 - 0.2))*0.95
    target_liquidity = (1-(exp(equity)/2 - 0.2))*0.05
    liquidity_top += 0.0025*cycle/12
    liquidity_bot = liquidity_top/2
    target = [target_equity, target_alternative, target_bond, target_liquidity]  # target risk contribution of equity, bond, alternative, liquidity
    liquidity_interval = (liquidity_bot, liquidity_top)  # liquidity proportion interval

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

    w_mkt, lam_mkt = MeanVariance(ER, Sig, rf, mu=0.3/12)
    w_BL, implied_confidence = BlackLitterman(w_riskparity, ER, Sig, lam_mkt, rf, tau, P, Q, cov)
    #w_BL, implied_confidence = BlackLitterman(w_riskparity, ER, cov, lam_mkt, rf, tau, P, Q, cov)
    risk_contribution_bl = RiskContribution(w_BL, cov)

    if output:
        print("risk parity weight")
        print(w_riskparity)
        print("risk parity risk contribution")
        print(risk_contribution)

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



fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(result_eq, label='Equally weighted')
ax.plot(result_rp, label='Risk parity')
ax.plot(result_mv, label='Mean variance')
ax.plot(result_bl, label='Risk adjusted black litterman')

names = ['equally weighted', 'risk parity', 'mean variance', 'black litterman']
results = [result_eq, result_rp, result_mv, result_bl]
table = pd.DataFrame(columns = ["SharpRatio", "MaximumDrawdown"])
for name, result in zip(names, results):
    SR = sharpe_ratio(result)
    MD = maximum_drawdown(result)
    table.loc[name] = [SR, MD]
print(table)

plt.plot(result_eq)
plt.plot(result_rp)
plt.plot(result_mv)
plt.plot(result_bl)

ax.legend()

date_str = list(result_bl.index)
date_tick = [datetime.datetime.strptime(str(i), '%m/%d/%Y') for i in list(result_bl.index)]
ax.set_xticks(date_str)

plt.show()
