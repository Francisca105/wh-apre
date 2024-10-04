from sklearn.preprocessing import MinMaxScaler

df_scaled = df.copy()

# scales the numeric variables to [0, 1]
df_scaled[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = MinMaxScaler().fit_transform(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

# Separate the features from the outcome
X_scaled = df_scaled.drop('target', axis=1)
y_scaled = df_scaled['target']

# Apply a stratified 5-fold with shuffling
folds_scaled = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Compute the cross validation accuracies for each model
knn_accs = cross_val_score(KNeighborsClassifier(), X_scaled, y_scaled, cv=folds_scaled, scoring='accuracy')
naive_accs = cross_val_score(GaussianNB(), X_scaled, y_scaled, cv=folds_scaled, scoring='accuracy')

print('kNN accuracy =', round(np.mean(knn_accs),2), "±", round(np.std(knn_accs),2))
print('naive Bayes accuracy =', round(np.mean(naive_accs),2), "±", round(np.std(naive_accs),2))