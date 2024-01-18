from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb

X, y = make_multilabel_classification(
    n_samples=32, n_classes=5, n_labels=3, random_state=0
)

clf = xgb.XGBClassifier(tree_method="hist")
clf = xgb.XGBClassifier(objective='binary:logistic', tree_method="hist")

clf.fit(X, y)
np.testing.assert_allclose(clf.predict(X), y)
accuracy_score(y, clf.predict(X))


probabilities = clf.predict_proba(X)
probabilities


#-------------------------------------------#
# trying to get multi-class classification to work

import pandas as pd

# Load the data
data = pd.read_csv('xgboost/winequality-red.csv')

# Display the first few rows of the data
data.head()

from sklearn.model_selection import train_test_split

# Separate target variable
X = data.drop('quality', axis=1)
y = data['quality'] - data['quality'].min()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier

# Create an instance of the XGBClassifier
model = XGBClassifier(objective='multi:softprob')

# Fit the model to the training data
model.fit(X_train, y_train)

# Make class predictions
y_pred = model.predict(X_test)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test)