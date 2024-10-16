import numpy as np

TRAIN,TEST = True,False
NON_REGULATED,REGULATED = True,False

def rmse(mode,model):
    accum = 0
    predict = 0 if model == NON_REGULATED else 1
    (start,stop) = (0,5) if mode == TRAIN else (5,8)
    for _ in range(start,stop):
                accum = accum + (z[_]-predicted[predict][_])**2

    return np.sqrt(accum/(stop-start))

z = [1.25,7.0,2.7,3.2,5.5,0.7,1.1,2.2]
non_regulated = [3.31595,0.11368]
regulated = [1.81805,0.32327]
phi = [1,3,6,9,8,4,2,5]
predicted_non_regulated = []
predicted_regulated = []
for _ in range(8):
    predicted_non_regulated.append(non_regulated[0] + non_regulated[1]*phi[_])
    predicted_regulated.append(regulated[0] + regulated[1]*phi[_])
predicted_non_regulated = [round(_,5) for _ in predicted_non_regulated]
predicted_regulated = [round(_,5) for _ in predicted_regulated]
print(predicted_regulated)
predicted = [predicted_non_regulated,predicted_regulated]
train_rmse_non_regulated = round(rmse(TRAIN,NON_REGULATED),5)
test_rmse_non_regulated = round(rmse(TEST,NON_REGULATED),5)
train_rmse_regulated = round(rmse(TRAIN,REGULATED),5)
test_rmse_regulated = round(rmse(TEST,REGULATED),5)
print('Non-regulated:')
print('Train:   ' + str(train_rmse_non_regulated))
print('Test:    ' + str(test_rmse_non_regulated))
print('Regulated:')
print('Train:   ' + str(train_rmse_regulated))
print('Test:    ' + str(test_rmse_regulated))