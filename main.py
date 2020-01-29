import BlackLitterman as BL
import numpy as np
from BlackLitterman import BlackLitterman, MeanVariance, MeanVarianceConstraint

######################### testing inputs, will be replaced
Sig = np.array([[1, 0.1, 0.1, 0.1],
                [0.1, 1, 0.1, 0.1],
                [0.1, 0.1, 1, 0.1],
                [0.1, 0.1, 0.1, 1]])
Sig /= 10

ER = np.array([0.01, 0.02, 0.03, 0.025]).reshape((-1, 1))
rf = 0.01

# Black Litterman Parameters
tau = 0.05  # 1 / n, n is # of observation

# View parameters
P = np.array([[0, 0, 1, -1],
              [-2, 1, 1, 0]])
Q = np.array([[0.01],
              [0.005]])
confidence = np.array([[0.5],
                       [0.4]])

#############################################


w_mkt, lam_mkt = MeanVariance(ER, Sig, rf)
print(BlackLitterman(w_mkt, Sig, lam_mkt, rf, tau, P, Q))
print(w_mkt)

print(MeanVarianceConstraint(Sig, ER, rf, lam_mkt))
