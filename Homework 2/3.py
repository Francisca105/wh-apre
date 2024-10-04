from scipy import stats 

res = stats.ttest_rel(accs[0], accs[1], alternative='greater')
print("Not scaled:\nkNN > naive Bayes?\npval=", round(res.pvalue, 3))

res = stats.ttest_rel(knn_accs, naive_accs, alternative='greater')
print("\nScaled:\nkNN > naive Bayes?\npval=", round(res.pvalue, 3))