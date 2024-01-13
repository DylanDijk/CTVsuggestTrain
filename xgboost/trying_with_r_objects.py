import rpy2.robjects as robjects
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb

robjects.r['load']('xgboost/train_features.RData')
robjects.r['load']('xgboost/train_res.RData')

train_features_py = robjects.r['train_features']
train_res_py = robjects.r['train_res']


# turn the R matrix into a numpy array
train_features_py = np.array(train_features_py)
train_features_py
train_res_py = np.array(train_res_py)
train_res_py


robjects.r['load']('xgboost/test_feature.RData')
robjects.r['load']('xgboost/test_res.RData')

test_features_py = robjects.r['test_feature']
test_res_py = robjects.r['test_res']

test_features_py = np.array(test_features_py)
test_features_py
test_res_py = np.array(test_res_py)
test_res_py

clf = xgb.XGBClassifier(tree_method="hist")
clf.fit(train_features_py, train_res_py)
np.testing.assert_allclose(clf.predict(test_features_py), test_res_py)
accuracy_score(test_res_py, clf.predict(test_features_py))

