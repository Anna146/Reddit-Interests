import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.set_printoptions(precision=3)

# params
parameters = {'n_estimators': [400, 500, 700],
              'criterion': ['gini', 'entropy'],
              'max_depth': [10, 20, 50],
              'bootstrap': [True],
              'class_weight': ['balanced', 'balanced_subsample'],
              'max_samples': [0.5, 0.3, 0.1]
            }

# models
clf = RandomForestClassifier(random_state=0)
cv = GridSearchCV(clf, parameters, scoring=None, n_jobs=-1, refit=False, cv=5, verbose=1, return_train_score=False)

# data
users = pickle.loads(open('./data/reddust_10profession_distance_df.pickle', 'rb').read())
X, y = users[list(range(0,11))], users["profession"]
X = StandardScaler().fit_transform(X)

# grid search
print("Grid search")
cv.fit(X, y)
best_params, best_score = [x for x in sorted(list(zip(cv.cv_results_['params'], cv.cv_results_['mean_test_score'])), key=lambda x:x[1], reverse=False) if not math.isnan(x[1])][-1]
print([x for x in sorted(list(zip(cv.cv_results_['params'], cv.cv_results_['mean_test_score'])), key=lambda x:x[1], reverse=False)])
print("Best grid search accuracy score", best_score)
print()

# cross
clf = RandomForestClassifier(random_state=33, **best_params, n_jobs=-1)
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
pickle.dump(clf, open("./data/checkpoints/random_forest.pkl", "wb"))




