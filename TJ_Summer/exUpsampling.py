import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

# Example imbalanced dataset
X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)

# Count the number of samples in each class
print("Before upsampling:")
print("Class distribution:", Counter(y))

# Apply RandomOverSampler to upsample the dataset
ros = RandomOverSampler(random_state=42)
X_upsampled, y_upsampled = ros.fit_resample(X, y)

# Count the number of samples in each class after upsampling
print("After upsampling:")
print("Class distribution:", Counter(y_upsampled))

# example of evaluating a decision tree with upsampled data
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold

# define pipeline
steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)

# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
score = mean(scores)
print('Mean ROC AUC: %.3f' % score)



