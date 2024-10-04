from sklearn import metrics
from sklearn.model_selection import train_test_split

k_values = [1, 5, 10, 20, 30]

# Splits the data in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train_accs, test_accs = [[] for _ in range(2)], [[] for _ in range(2)]
weights = ['uniform', 'distance']

# Computes the accuracy of each model for each k neighbors
for k in k_values:
    i = 0
    for weight in weights:
        predictor = KNeighborsClassifier(n_neighbors=k, weights=weight)
        predictor.fit(X_train, y_train)
        train_accs[i].append(metrics.accuracy_score(y_train, predictor.predict(X_train)))
        test_accs[i].append(metrics.accuracy_score(y_test, predictor.predict(X_test)))
        i = i + 1       

# Plots the accuracies graphs
i = 0
for weight in weights:
    plt.plot(k_values, train_accs[i], 'o-', label='Training accuracy')
    plt.plot(k_values, test_accs[i], 'o-', label='Test accuracy')
    plt.title(f'Training/Testing accuracy with {weight} weights')
    plt.xlabel('k Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    i = i + 1