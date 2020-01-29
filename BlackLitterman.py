import numpy as np
import pandas as pd

######################### testing inputs, will be replaced
Sig = np.array([[1, 0.1, 0.1, 0.1],
                [0.1, 1, 0.1, 0.1],
                [0.1, 0.1, 1, 0.1],
                [0.1, 0.1, 0.1, 1]])

mu = np.array([0.01, 0.02, 0.03, 0.025]).reshape((-1, 1))

lam = 1

w_blInput = np.array([0.1, 0.4, 0.3, 0.2]).reshape((-1, 1))

########################

# Black Litterman Parameters

tau = 0.05


# View parameters

P = np.array([[0, 0, 1, -1]])

Q = np.array([[0.01]])

confidence = 0.5
Omeg = np.array([[1]])


# Computation

Pi = lam * Sig.dot(w_blInput)

mu_BL_1 = np.linalg.inv(tau * Sig) + P.T.dot(Omeg).dot(P)
mu_BL_2 = np.linalg.inv(tau * Sig).dot(mu) + P.T.dot(Omeg).dot(Q)
mu_BL = np.linalg.inv(mu_BL_1).dot(mu_BL_2)

#mu_BL100 = Pi + tau * S

Sig_BL = np.linalg.inv(mu_BL_1)


# New mean variance analysis
from scipy.optimize import minimize



# equity 的比重要保证 subject to