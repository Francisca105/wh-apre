import numpy as np

X = np.matrix([[1,1],[1,3],[1,6],[1,9],[1,8]])
z = np.matrix([[1.25],[7.0],[2.7],[3.2],[5.5]])
X_cross = np.linalg.pinv(X).round(5)
w = np.matmul(X_cross,z).round(5)
print(w)