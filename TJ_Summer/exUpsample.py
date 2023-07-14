import numpy as np
import imblearn.over_sampling as over_sampling
import imblearn.under_sampling as under_sampling
import imblearn.combine as combine
import imblearn.ensemble as ensemble
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(1000)

nb_samples = 5000

# Create the dataset
X, Y = make_classification(n_samples=nb_samples,
                            n_features=2,
                            n_informative=2,
                            n_redundant=0,
                            n_classes=2,
                            n_clusters_per_class=1,
                            weights=[0.99, 0.01],
                            flip_y=0.01)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Show the original distribution
plt.figure(figsize=(15, 7))
plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
plt.title('Original dataset')

# Oversample the dataset
ros = over_sampling.RandomOverSampler(random_state=1000)
X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

# Show the new distribution
plt.subplot(122)
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=Y_resampled)
plt.title('Oversampled dataset')
plt.show()

# Undersample the dataset
rus = under_sampling.RandomUnderSampler(random_state=1000)
X_resampled, Y_resampled = rus.fit_resample(X_train, Y_train)

# Show the new distribution
plt.figure(figsize=(15, 7))
plt.subplot(121)
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=Y_resampled)
plt.title('Undersampled dataset')

# Combine over- and under-sampling
cc = combine.SMOTEENN(random_state=1000)
X_resampled, Y_resampled = cc.fit_resample(X_train, Y_train)

# Show the new distribution
plt.subplot(122)
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=Y_resampled)
plt.title('Combined dataset')
plt.show()
