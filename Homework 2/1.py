import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

# Read the csv dataset
df = pd.read_csv('heart-disease.csv', delimiter=',')

# Separate the features from the outcome
X = df.drop('target', axis=1)
y = df['target']

# Apply a stratified 5-fold with shuffling
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Compute the cross validation accuracies for each model
accs = [cross_val_score(KNeighborsClassifier(), X, y, cv=folds, scoring='accuracy'),
        cross_val_score(GaussianNB(), X, y, cv=folds, scoring='accuracy')]

# Plots the accuracy boxplot for both models
labels = ['kNN', 'naive Bayes']
colors = ['peachpuff', 'orange']

fig, ax = plt.subplots()
ax.set_ylabel('Accuracy')

bplot = ax.boxplot(accs, patch_artist=True, tick_labels=labels)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()