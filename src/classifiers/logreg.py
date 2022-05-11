import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.set_printoptions(precision=3)

# params
parameters = {"class_weight" : [None, "balanced"],
              "penalty" : ['l1', 'l2', 'elasticnet', None],
              'C': [0.1 * x for x in range(1, 11)],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'tol':[0.001, 0.01, 0.1, 0.0001]}

# models
clf = LogisticRegression(random_state=33, max_iter=1000)
cv = GridSearchCV(clf, parameters, scoring=None, n_jobs=-1, refit=False, cv=5, verbose=1, return_train_score=False)

# data
users = pickle.loads(open('./data/reddust_10profession_distance_df.pickle', 'rb').read())
X, y = users[list(range(0,11))], users["profession"]
X = StandardScaler().fit_transform(X)

# grid search
print("Grid search")
cv.fit(X, y)
best_params, best_score = [x for x in sorted(list(zip(cv.cv_results_['params'], cv.cv_results_['mean_test_score'])), key=lambda x:x[1], reverse=False) if not math.isnan(x[1])][-1]
print("Best grid search accuracy score", best_score)
print()

# cross
clf = LogisticRegression(random_state=33, max_iter=1000, **best_params)
scores = cross_validate(clf, X, y, cv=10, scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'))
print("Cross val results")
print('accuracy', scores['test_accuracy'].mean())
print('accuracy stdev', scores['test_accuracy'].std())
print('precision', scores['test_precision_macro'].mean())
print('precision stdev', scores['test_precision_macro'].std())
print('recall', scores['test_recall_macro'].mean())
print('recall stdev', scores['test_recall_macro'].std())
print('f1', scores['test_f1_macro'].mean())
print('f1 stdev', scores['test_f1_macro'].std())
#pickle.dump(clf, open("./data/checkpoints/logreg.pkl", "wb"))



