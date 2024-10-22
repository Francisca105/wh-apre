import numpy as np
import scipy.stats as stats

def update_u(X,probs,k):
    accum = [0,0]
    for i in range(3):
        accum[0] = accum[0] + X[i][0]*probs[i][k]
        accum[1] = accum[1] + X[i][1]*probs[i][k]
    res = [(_/sum([__[k] for __ in probs])).round(3) for _ in accum]
    return res

def update_E(X,probs,k,u):
    accum = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            for _ in range(3):
                accum[i][j] = accum[i][j] + probs[_][k]*(X[_][i]-u[i])*(X[_][j]-u[j])
    res = [[(__/sum([___[k] for ___ in probs])).round(3) for __ in _] for _ in accum]
    return res


X1 = [1,0]
X2 = [0,2]
X3 = [3,-1]
X = [X1,X2,X3]
u1 = [2,-1]
u2 = [1,1]
u = [u1,u2]
sigma1 = [[2,1],[1,2]]
sigma2 = [[2,0],[0,2]]
pi1 = 0.5
pi2 = 0.5

# Epoch 1
print('Epoch 1\n')

# E-step
print('E-step:\n')

# X1
print('X1:')
probs_X1 = [(stats.multivariate_normal(u1, sigma1).pdf(X1).round(3) * pi1).round(3), (stats.multivariate_normal(u2, sigma2).pdf(X1).round(3) * pi2).round(3)]
print(f'N(X1|u=u1,E=E1)={(stats.multivariate_normal(u1,sigma1).pdf(X1)).round(3)};\tpi1={pi1};\tN(X1|u=u1,E=E1)*pi1={probs_X1[0]}')
print(f'N(X1|u=u2,E=E2)={(stats.multivariate_normal(u2,sigma2).pdf(X1)).round(3)};\tpi2={pi2};\tN(X1|u=u2,E=E2)*pi2={probs_X1[1]}')
probs_X1_norm = [(_/sum(probs_X1)).round(3) for _ in probs_X1]
print(f'P(k=1|X1) = {probs_X1[0]}/{probs_X1[0]}+{probs_X1[1]} = {probs_X1_norm[0]}')
print(f'P(k=2|X1) = {probs_X1[1]}/{probs_X1[0]}+{probs_X1[1]} = {probs_X1_norm[1]}\n')

# X2
print('X2')
probs_X2 = [(stats.multivariate_normal(u1, sigma1).pdf(X2).round(3) * pi1).round(3), (stats.multivariate_normal(u2, sigma2).pdf(X2).round(3) * pi2).round(3)]
print(f'N(X2|u=u1,E=E1)={(stats.multivariate_normal(u1,sigma1).pdf(X2)).round(3)};\tpi1={pi1};\tN(X2|u=u1,E=E1)*pi1={probs_X2[0]}')
print(f'N(X2|u=u2,E=E2)={(stats.multivariate_normal(u2,sigma2).pdf(X2)).round(3)};\tpi2={pi2};\tN(X2|u=u2,E=E2)*pi2={probs_X2[1]}')
probs_X2_norm = [(_/sum(probs_X2)).round(3) for _ in probs_X2]
print(f'P(k=1|X2) = {probs_X2[0]}/{probs_X2[0]}+{probs_X2[1]} = {probs_X2_norm[0]}')
print(f'P(k=1|X2) = {probs_X2[1]}/{probs_X2[0]}+{probs_X2[1]} = {probs_X2_norm[1]}\n')

# X3
print('X3')
probs_X3 = [(stats.multivariate_normal(u1, sigma1).pdf(X3).round(3) * pi1).round(3), (stats.multivariate_normal(u2, sigma2).pdf(X3).round(3) * pi2).round(3)]
print(f'N(X3|u=u1,E=E1)={(stats.multivariate_normal(u1,sigma1).pdf(X3)).round(3)};\tpi1={pi1};\tN(X3|u=u1,E=E1)*pi1={probs_X3[0]}')
print(f'N(X3|u=u2,E=E2)={(stats.multivariate_normal(u2,sigma2).pdf(X3)).round(3)};\tpi2={pi2};\tN(X3|u=u2,E=E2)*pi2={probs_X3[1]}\n')
probs_X3_norm = [(_/sum(probs_X3)).round(3) for _ in probs_X3]
print(f'P(k=1|X3) = {probs_X3[0]}/{probs_X3[0]}+{probs_X3[1]} = {probs_X3_norm[0]}')
print(f'P(k=1|X3) = {probs_X3[1]}/{probs_X3[0]}+{probs_X3[1]} = {probs_X3_norm[1]}\n')

probs_X_norm = [probs_X1_norm,probs_X2_norm,probs_X3_norm]

# M-step
print('M-step:\n')

# Cluster 1
print('Cluster 1')

# u
u1_updated = [float(_) for _ in update_u(X,probs_X_norm,0)]
print(f'u1 = {u1_updated}')

# E
E1_updated = [[float(__) for __ in _] for _ in update_E(X,probs_X_norm,0,u1_updated)]
print(f'E1 = {E1_updated}')

# P(k)
prob_k1 = (sum([_[0] for _ in probs_X_norm])/3).round(3)
print(f'P(k=1) = {prob_k1}\n')

# Cluster 2
print('Cluster 2')

# u
u2_updated = [float(_) for _ in update_u(X,probs_X_norm,1)]
print(f'u2 = {u2_updated}')

# E
E2_updated = [[float(__) for __ in _] for _ in update_E(X,probs_X_norm,1,u2_updated)]
print(f'E2 = {E2_updated}')

# P(k)
prob_k2 = (sum([_[1] for _ in probs_X_norm])/3).round(3)
print(f'P(k=2) = {prob_k2}\n')