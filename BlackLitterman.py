import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize, LinearConstraint

######################### testing inputs, will be replaced
Sig = np.array([[1, 0.1, 0.1, 0.1],
                [0.1, 1, 0.1, 0.1],
                [0.1, 0.1, 1, 0.1],
                [0.1, 0.1, 0.1, 1]])

ER = np.array([0.01, 0.02, 0.03, 0.025]).reshape((-1, 1))
rf = 0.01

def MeanVariance(Sig, ER, rf):
    # compute market portfolio
    l = np.ones((Sig.shape[0], 1))
    A = l.T.dot(inv(Sig)).dot(ER)
    B = ER.T.dot(inv(Sig)).dot(ER)
    C = l.T.dot(inv(Sig)).dot(l)

    w_mkt = inv(Sig).dot(ER - rf * l) / (A - C * rf)
    mu = ER.T.dot(w_mkt)
    lam = (mu - rf) / (ER - rf * l).T.dot(inv(Sig)).dot(ER - rf * l)
    # w_capm = lam * inv(Sig).dot(ER - rf * l)

    return w_mkt, lam

# input from riskparity
w_blInput = np.array([0.1, 0.4, 0.3, 0.2]).reshape((-1, 1))

w_blInput, lam_mkt = MeanVariance(Sig, ER, rf)
########################

# Black Litterman Parameters

tau = 0.05 # 1 / n, n is # of observation

# View parameters

P = np.array([[0, 0, 1, -1],
              [-2, 1, 1, 0]])

Q = np.array([[0.01],
              [0.005]])


confidence = np.array([[0.5],
                       [0.4]])

Omeg = np.diag((P.dot(Sig).dot(P.T) * tau).diagonal())


# Computation
Pi = lam_mkt * Sig.dot(w_blInput)

ER_BL_1 = inv(tau * Sig) + P.T.dot(inv(Omeg)).dot(P)
ER_BL_2 = inv(tau * Sig).dot(Pi) + P.T.dot(inv(Omeg)).dot(Q)
ER_BL = inv(ER_BL_1).dot(ER_BL_2)
Sig_BL = Sig + inv(ER_BL_1)

# 100 percent confidence
ER_BL100 = Pi + tau * Sig.dot(P.T).dot(inv(P.dot(tau * Sig).dot(P.T))).dot(Q - P.dot(Pi))
Sig_BL100 = Sig



# New mean variance analysis
w_BL, lam_BL = MeanVariance(Sig_BL, ER_BL, rf)
w_BL100, lam_BL100 = MeanVariance(Sig_BL100, ER_BL100, rf)
'''
# tiltering methods
if 0:
    tilt = (w_BL100 - w_blInput) * confidence
    w_tilt = w_blInput + tilt
    for i in range(P.shape[0]):
        ER_tmp = Pi + tau * Sig.dot(P[i].T).dot(inv(P[i].dot(tau * Sig).dot(P[i].T))).dot(Q[i] - P[i].dot(Pi))
        w_tmp = inv(lam_mkt * Sig).dot(ER_tmp)
        departure_tmp = w_tmp - w_blInput
'''

print(w_BL, w_BL100)
print(ER_BL, ER_BL100)
print(np.abs(w_BL - w_blInput).sum() / np.abs(w_BL100 - w_blInput).sum())


# numerical mean variance analysis
def lossfunc(x):
    x = x.T
    l = np.ones(shape=(x.shape[0], 1))
    print(x)
    return -(x.T.dot(ER - rf * l) - 1/lam_mkt * x.T.dot(Sig).dot(x) / 2)
re = minimize(lossfunc, w_blInput.T)
print(re)

# equity 的比重要保证 subject to
pass


