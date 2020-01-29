import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize, LinearConstraint


def MeanVariance(ER, Sig, rf):
    ER = np.array(ER).reshape((-1, 1))
    Sig = np.array(Sig)
    # compute market portfolio
    l = np.ones((Sig.shape[0], 1))
    A = l.T.dot(inv(Sig)).dot(ER)
    B = ER.T.dot(inv(Sig)).dot(ER)
    C = l.T.dot(inv(Sig)).dot(l)

    w_mkt = inv(Sig).dot(ER - rf * l) / (A - C * rf)
    mu = ER.T.dot(w_mkt)
    lam = (mu - rf) / w_mkt.T.dot(Sig).dot(w_mkt) #1 / ((mu - rf) / (ER - rf * l).T.dot(inv(Sig)).dot(ER - rf * l)) # (mu - rf) / w_mkt.T.dot(Sig).dot(w_mkt)
    # w_capm = lam * inv(Sig).dot(ER - rf * l)

    return w_mkt, lam


def BlackLitterman(w_blInput, Sig, lam, rf, tau, P, Q):

    Sig = np.array(Sig)

    # Computation
    Pi = lam * Sig.dot(w_blInput)

    Omeg = np.diag((P.dot(Sig).dot(P.T) * tau).diagonal())

    ER_BL_1 = inv(tau * Sig) + P.T.dot(inv(Omeg)).dot(P)
    ER_BL_2 = inv(tau * Sig).dot(Pi) + P.T.dot(inv(Omeg)).dot(Q)
    ER_BL = inv(ER_BL_1).dot(ER_BL_2)
    Sig_BL = tau * Sig + inv(ER_BL_1)

    # 100 percent confidence
    ER_BL100 = Pi + tau * Sig.dot(P.T).dot(inv(P.dot(tau * Sig).dot(P.T))).dot(Q - P.dot(Pi))
    Sig_BL100 = tau * Sig


    # New mean variance analysis
    w_BL, lam_BL = MeanVariance(ER_BL, Sig_BL, rf)  # new weight
    w_BL100, lam_BL100 = MeanVariance(ER_BL100, Sig_BL100, rf) # 100 confidence weight

    '''
    remain to be completed
    # tiltering methods
    if 0:
        tilt = (w_BL100 - w_blInput) * confidence
        w_tilt = w_blInput + tilt
        for i in range(P.shape[0]):
            ER_tmp = Pi + tau * Sig.dot(P[i].T).dot(inv(P[i].dot(tau * Sig).dot(P[i].T))).dot(Q[i] - P[i].dot(Pi))
            w_tmp = inv(lam_mkt * Sig).dot(ER_tmp)
            departure_tmp = w_tmp - w_blInput
    '''
    w_BL = inv(lam * Sig_BL).dot(ER_BL)
    w_BL100 = inv(lam * Sig_BL100).dot(ER_BL100)

    implied_confidence = np.abs(w_BL - w_blInput).sum() / np.abs(w_BL100 - w_blInput).sum()

    return w_BL, implied_confidence


def MeanVarianceConstraint(Sig, ER, rf, lam):
    '''
    Numerical Mean Variance Optimizer, taking into account constraints
    '''
    from scipy.optimize import minimize, LinearConstraint

    # object function
    def objfunc(x):
        x = x.T
        l = np.ones(shape=(x.shape[0], 1))
        return -(x.T.dot(ER - rf * l) - lam * x.T.dot(Sig).dot(x) / 2)

    # params of optimizer
    x0 = np.zeros(ER.shape)
    bnds = tuple((None,None) for _ in x0)
    cons = (#{'type':'eq', 'fun': lambda x: sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x})
    options={'disp':False, 'maxiter':1000, 'ftol':1e-20}
    re = minimize(objfunc, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    re = minimize(objfunc, x0)
    return re

    # equity 的比重要保证 subject to


