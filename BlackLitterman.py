import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def MeanVariance(ER, Sig, rf, mu=0.3 / 12, plot=False):
    ER = np.array(ER).reshape((-1, 1))
    Sig = np.array(Sig)

    # compute market portfolio
    l = np.ones((Sig.shape[0], 1))
    A = l.T.dot(inv(Sig)).dot(ER)
    B = ER.T.dot(inv(Sig)).dot(ER)
    C = l.T.dot(inv(Sig)).dot(l)
    D = B * C - A ** 2
    H = B - 2 * rf * A + rf ** 2 * C
    w_mkt = inv(Sig).dot(ER - rf * l) / (A - C * rf)
    mu_mkt = ER.T.dot(w_mkt)
    lam = (mu_mkt - rf) / w_mkt.T.dot(Sig).dot(
        w_mkt)

    # portfolio weight when specify expected return
    alpha = ((mu - rf) / (ER - rf * l).T.dot(inv(Sig)).dot(ER - rf * l))

    w_mu = alpha * inv(Sig).dot(ER - rf * l)
    w_mu = inv(Sig).dot(1 / D * (B - mu * A) * l + 1 / D * (mu * C - A) * ER)
    w_mu0 = 1 - alpha * (A - C * rf)

    if plot:
        x = np.arange(0, 0.02, 0.001)
        y = []
        for mu in x:
            alpha = ((mu - rf) / (ER - rf * l).T.dot(inv(Sig)).dot(ER - rf * l))
            w_mu = inv(Sig).dot(1 / D * (B - mu * A) * l + 1 / D * (mu * C - A) * ER)
            y.append(w_mu.T.dot(Sig).dot(w_mu)[0])
        y = np.array(y)
        y = np.sqrt(y)
        plt.plot(y, x)
        plt.show()

    return w_mu, lam


def MeanVarianceConstraint(ER, Sig, rf):
    from scipy.optimize import minimize

    # compute the unconstraint market portfolio and lam
    l = np.ones(shape=(ER.shape[0], 1))
    A = l.T.dot(inv(Sig)).dot(ER)
    C = l.T.dot(inv(Sig)).dot(l)
    w_mkt = inv(Sig).dot(ER - rf * l) / (A - C * rf)
    mu_mkt = ER.T.dot(w_mkt)
    lam = (mu_mkt - rf) / w_mkt.T.dot(Sig).dot(w_mkt)

    mu = 0.5 / 12
    # object function (use unconstraint lam)
    def objfunc(x):
        x = np.array(x).T
        l = np.ones(shape=(x.shape[0], 1))
        return x.T.dot(Sig).dot(x) / 2

    # params of optimizer
    x0 = np.ones(ER.shape)
    x0 /= x0.sum()  # start with equally weighted
    bnds = tuple((0, None) for _ in x0)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},  # sum to 1
            {'type': 'ineq', 'fun': lambda x: 0.8 - sum(x[0:3]) / x.sum()})  # equity limit

    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},  # sum to 1
            {'type': 'ineq', 'fun': lambda x: 0.8 - sum(x[0:3]) / x.sum()},
            {'type': 'eq', 'fun': lambda x: x.T.dot(ER) - mu})  # equity limit
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

    re = minimize(objfunc, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    wts = np.array(re.x)
    return wts


def BlackLitterman(w_blInput, ER, Sig, lam, rf, tau, P, Q, cov):
    Sig = np.array(Sig)
    w_blInput = np.array(w_blInput).reshape((-1, 1))

    # Computation
    Pi = lam * Sig.dot(w_blInput)
    Pi = 1 * Sig.dot(w_blInput)
    # Pi = np.array(ER).reshape((-1, 1))

    Omeg = np.diag((P.dot(Sig).dot(P.T) * 0.0001).diagonal())

    ER_BL_1 = inv(tau * Sig) + P.T.dot(inv(Omeg)).dot(P)
    ER_BL_2 = inv(tau * Sig).dot(Pi) + P.T.dot(inv(Omeg)).dot(Q)
    ER_BL = inv(ER_BL_1).dot(ER_BL_2)
    Sig_BL = tau * Sig + inv(ER_BL_1)

    # 100 percent confidence
    ER_BL100 = Pi + tau * Sig.dot(P.T).dot(inv(P.dot(tau * Sig).dot(P.T))).dot(Q - P.dot(Pi))
    Sig_BL100 = tau * Sig

    # New mean variance analysis
    w_BL, lam_BL = MeanVariance(ER_BL, Sig_BL, rf)  # new weight
    w_BL = MeanVarianceConstraint(ER_BL, Sig_BL, rf)  # new weight

    target = [0.7, 0.198, 0.1, 0.002]  # target risk contribution of equity, bond, alternative, liquidity
    equity = 0.8  # equity proportion limit
    liquidity_interval = (0.05, 0.1)  # liquidity proportion interval
    # w_BL = np.array(risk_parity_weight(Sig_BL, target, equity, liquidity_interval))
    w_BL100, lam_BL100 = MeanVariance(ER_BL100, Sig_BL100, rf)  # 100 confidence weight

    # w_BL = inv(lam * Sig_BL).dot(ER_BL)
    # w_BL100 = inv(lam * Sig_BL100).dot(ER_BL100)

    implied_confidence = np.abs(w_BL - w_blInput).sum() / np.abs(w_BL100 - w_blInput).sum()

    # new risk contribution
    var = (w_BL.T).dot(Sig).dot(w_BL)
    risk_contribution = np.array(w_BL * (Sig.dot(w_BL)) / var)
    risk_contribution = [risk_contribution[0:3].sum(), risk_contribution[3:5].sum(), risk_contribution[5:9].sum()]

    return w_BL, implied_confidence



