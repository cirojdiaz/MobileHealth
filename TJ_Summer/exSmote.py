import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

sns.set_style("darkgrid")
sns.set_context("poster")
plt.rcParams["figure.figsize"] = [8,6]

#import file
spam_dataset = pd.read_csv("spam.csv", encoding = 'latin')
spam_dataset = spam_dataset[["v1", "v2"]]
spam_dataset.head()

print("Before Upsampling:")
print(spam_dataset["v1"].value_counts())

#convert text to numbers
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(spam_dataset ['v2'])

#convert labels to numbers
spam_dataset['v1'] = spam_dataset['v1'].map({'ham': 0, 'spam': 1})
spam_dataset.head()

#extract label set
y = spam_dataset[['v1']]

#Use SMOTE for upsampling
su = SMOTE(random_state=42)
X_su, y_su = su.fit_resample(X, y)

print("After Upsampling:")
print(y_su["v1"].value_counts())

y_su.groupby('v1').size().plot(kind='pie',
                                       y = "v1",
                                       label = "Type",
                                       autopct='%1.1f%%')
plt.show()

for k in trends.keys():
    age_values = list(trends[k].keys())
    data_values = [k for i, k in trends[k].items()]
    plt.plot(age_values, data_values, label="Data")
    
    # Calculate moving average
    window = 5
    moving_avg = []
    for i in range(len(data_values)):
        if i < window:
            window_data = data_values[:i+1]
        else:
            window_data = data_values[i-window+1:i+1]
        avg = sum(window_data) / len(window_data)
        moving_avg.append(avg)