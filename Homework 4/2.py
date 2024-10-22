import numpy as np
import scipy.stats as stats
import math

def round(number, ndigits=0):
    """Always round off"""
    exp = number * 10 ** ndigits
    if abs(exp) - abs(math.floor(exp)) < 0.5:
        return type(number)(math.floor(exp) / 10 ** ndigits)
    return type(number)(math.ceil(exp) / 10 ** ndigits)

X1 = [1,0]
X2 = [0,2]
X3 = [3,-1]
u1 = [2.455,-0.718]
u2 = [0.503,1.111]
E1 = [[0.812,-0.43],[-0.43,0.24]]
E2 = [[0.487,-0.678],[-0.678,1.106]]
prob_k1 = 0.425
prob_k2 = 0.575

print('X1:')
probs_X1 = [round(round(stats.multivariate_normal(u1, E1).pdf(X1), 3) * prob_k1, 3), round(round(stats.multivariate_normal(u2, E2).pdf(X1), 3) * prob_k2, 3)]
print(f'N(X1|u=u1,E=E1)={round(stats.multivariate_normal(u1,E1).pdf(X1), 3)};\tP(k=1)={prob_k1};\tN(X1|u=u1,E=E1)*P(k=1)={probs_X1[0]}')
print(f'N(X1|u=u2,E=E2)={round(stats.multivariate_normal(u2,E2).pdf(X1), 3)};\tP(k=2)={prob_k2};\tN(X1|u=u2,E=E2)*P(k=2)={probs_X1[1]}\n')

print('X2:')
probs_X2 = [round(round(stats.multivariate_normal(u1, E1).pdf(X2), 3) * prob_k1, 3), round(round(stats.multivariate_normal(u2, E2).pdf(X2), 3) * prob_k2, 3)]
print(f'N(X2|u=u1,E=E1)={round(stats.multivariate_normal(u1,E1).pdf(X2), 3)};\tP(k=1)={prob_k1};\tN(X1|u=u1,E=E1)*P(k=1)={probs_X2[0]}')
print(f'N(X2|u=u2,E=E2)={round(stats.multivariate_normal(u2,E2).pdf(X2), 3)};\tP(k=2)={prob_k2};\tN(X1|u=u2,E=E2)*P(k=2)={probs_X2[1]}\n')

print('X3:')
probs_X3 = [round(round(stats.multivariate_normal(u1, E1).pdf(X3), 3) * prob_k1, 3), round(round(stats.multivariate_normal(u2, E2).pdf(X3), 3) * prob_k2, 3)]
print(f'N(X3|u=u1,E=E1)={round(stats.multivariate_normal(u1,E1).pdf(X3), 3)};\tP(k=1)={prob_k1};\tN(X1|u=u1,E=E1)*P(k=1)={probs_X3[0]}')
print(f'N(X3|u=u2,E=E2)={round(stats.multivariate_normal(u2,E2).pdf(X3), 3)};\tP(k=2)={prob_k2};\tN(X1|u=u2,E=E2)*P(k=2)={probs_X3[1]}')