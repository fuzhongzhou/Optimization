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


w_mkt, lam = MeanVariance(Sig, ER, rf)
########################

# Black Litterman Parameters

tau = 0.05 # 1 / n n is # of observation

# View parameters

P = np.array([[0, 0, 1, -1]])

Q = np.array([[0.01]])

confidence = 0.5
Omeg = np.array([[1]])


# Computation
Pi = lam * Sig.dot(w_blInput)

ER_BL_1 = inv(tau * Sig) + P.T.dot(Omeg).dot(P)
ER_BL_2 = inv(tau * Sig).dot(ER) + P.T.dot(Omeg).dot(Q)
ER_BL = inv(ER_BL_1).dot(ER_BL_2)
Sig_BL = inv(ER_BL_1)

# 100 percent confidence
mu_BL100 = Pi + tau * Sig.dot(P.T).dot(inv(P.dot(tau * Sig).dot(P.T))).dot(Q - P.dot(Pi))
Sig_BL100 = inv(inv(tau * Sig) + P.T.dot(np.zeros(shape=Omeg.shape)).dot(P))



# New mean variance analysis

def lossfunc(x):
    x = x.T
    l = np.ones(shape=(x.shape[0], 1))
    print(x)
    return -(x.T.dot(ER - rf * l) - 1/lam * x.T.dot(Sig).dot(x) / 2)
re = minimize(lossfunc, w_mkt.T)
print(re)

# equity 的比重要保证 subject to
pass


