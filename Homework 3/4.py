import numpy as np
from scipy.special import softmax

t = np.matrix([[0],[1],[0]])
lr = 0.1
X0 = np.matrix([[1],[1]])
W1 = np.matrix([[0.1,0.1],[0.1,0.2],[0.2,0.1]])
b1 = np.matrix([[0.1],[0],[0.1]])
W2 = np.matrix([[1,2,2],[1,2,1],[1,1,1]])
b2 = np.matrix([[1],[1],[1]])

# Forward Propagation
Z1 = np.matmul(W1,X0) + b1
X1 = Z1

Z2 = np.matmul(W2,X1) + b2

X2 = softmax(Z2).round(3)

# Backward Propagation

delta2 = X2 - t
W2 = (W2 - lr*(np.matmul(delta2,np.transpose(X1)))).round(3)
b2 = (b2 - lr*delta2).round(3)

delta1 = np.matmul(np.transpose(W2),delta2).round(3)
W1 = (W1 - lr*(np.matmul(delta1,np.transpose(X0)))).round(3)
b1 = (b1 - lr*delta1).round(3)
print(np.array_str(b1, suppress_small=True))