import numpy as np

X_transpose = np.matrix([[1,1,1,1,1],[1,3,6,9,8]])
z = np.matrix([[1.25],[7.0],[2.7],[3.2],[5.5]])
X_transpose_X_plus_lambda_I = np.matrix([[6,27],[27,192]])
X_transpose_X_plus_lambda_I_inverse = np.linalg.inv(X_transpose_X_plus_lambda_I).round(5)
X_transpose_X_plus_lambda_I_inverse_X_transpose = np.matmul(X_transpose_X_plus_lambda_I_inverse,X_transpose).round(5)
w = np.matmul(X_transpose_X_plus_lambda_I_inverse_X_transpose,z).round(5)
print(w)