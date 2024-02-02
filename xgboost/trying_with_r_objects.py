import rpy2.robjects as robjects
import numpy as np
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb

# loading R objects
robjects.r['load']('xgboost/train_features.RData')
robjects.r['load']('xgboost/train_res.RData')
robjects.r['load']('xgboost/test_feature.RData')
robjects.r['load']('xgboost/test_res.RData')

robjects.r['load']('xgboost/predict_prob.Rdata')

train_features_py = robjects.r['train_features']
train_res_py = robjects.r['train_res']
test_features_py = robjects.r['test_feature']
test_res_py = robjects.r['test_res']
predict_prob = robjects.r['predict_prob']

# turn the R matrix into a numpy array
train_features_py = np.array(train_features_py)
train_res_py = np.array(train_res_py)

test_features_py = np.array(test_features_py)
test_res_py = np.array(test_res_py)
predict_prob = np.array(predict_prob)

# get dimension
train_res_py.shape


#-----------------------------------------------------------------------#
# Multi-class classification accuracy
# Now first want to compare xgboost classification accuracy for multi-class classification
# From the multinomial model we get: 80.2% accuracy

# Now looking at xgboost accuracy:
clf = xgb.XGBClassifier(objective='multi:softprob')

# To get multi-class accuracy, I have removed observations with multiple labels
multi_label_rows = np.zeros(train_res_py.shape[0])
for i in range(0, train_res_py.shape[0]):
    if sum(train_res_py[i,]) > 1:
        multi_label_rows[i] = 1

train_res_sing = train_res_py[multi_label_rows == 0,]
train_features_sing = train_features_py[multi_label_rows == 0,]  

multi_label_rows = np.zeros(test_res_py.shape[0])
for i in range(0, test_res_py.shape[0]):
    if sum(test_res_py[i,]) > 1:
        multi_label_rows[i] = 1

test_res_sing = test_res_py[multi_label_rows == 0,]
test_features_sing = test_features_py.transpose()[multi_label_rows == 0,]

# change train_res_sing to give the class of the multilabel observation
train_res_class = np.zeros(train_res_sing.shape[0])
for i in range(0, train_res_sing.shape[0]):
    train_res_class[i] = np.where(train_res_sing[i,]==1)[0][0]

test_res_class = np.zeros(test_res_sing.shape[0])
for i in range(0, test_res_sing.shape[0]):
    test_res_class[i] = np.where(test_res_sing[i,]==1)[0][0]

clf.fit(train_features_sing, train_res_class)
clf.score(test_features_sing, test_res_class)
# get 85% accuracy, with much faster training time
# but this is with removing all observations with multiple labels, so is not an exact comparison.


# Now with exact comparison:
# Need to repeat rows with multiple labels

# Gives vector with the number of labels for each observation
multi_label_rows = np.zeros(train_res_py.shape[0]).astype(int)
for i in range(0, train_res_py.shape[0]):
    multi_label_rows[i] = sum(train_res_py[i,])

train_feat_repeat = np.repeat(train_features_py, multi_label_rows, axis=0)

# function that expands binary labels into multiple rows for response matrix
def expand_binary_labels(array):
    expanded_rows = []

    for row in array:
        unique_labels = np.where(row == 1)[0]

        if len(unique_labels) > 1:
            # Row has multiple labels, create copies and set each label to 1
            for label in unique_labels:
                new_row = np.zeros_like(row)
                new_row[label] = 1
                expanded_rows.append(new_row)
        else:
            # Row has a single label or none, append as is
            expanded_rows.append(row)

    expanded_array = np.array(expanded_rows)

    return expanded_array

# Example usage:
binary_array = np.array([[1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [1, 1, 0, 0],
                        [0, 0, 1, 0]])

expanded_array = expand_binary_labels(binary_array)
print(expanded_array)

train_res_repeat = expand_binary_labels(train_res_py)

train_feat_repeat.shape
train_res_repeat.shape

train_res_class = np.zeros(train_res_repeat.shape[0])
for i in range(0, train_res_repeat.shape[0]):
    train_res_class[i] = np.where(train_res_repeat[i,]==1)[0][0]

clf.fit(train_feat_repeat, test_res_class)

multi_label_rows = np.zeros(test_res_py.shape[0]).astype(int)
for i in range(0, test_res_py.shape[0]):
    multi_label_rows[i] = sum(test_res_py[i,])

test_feat_repeat = np.repeat(test_features_py.transpose(), multi_label_rows, axis=0)
test_res_repeat = expand_binary_labels(test_res_py)
test_res_class = np.zeros(test_res_repeat.shape[0])
for i in range(0, test_res_repeat.shape[0]):
    test_res_class[i] = np.where(test_res_repeat[i,]==1)[0][0]
clf.score(test_feat_repeat, test_res_class)
# can see that now we acctually get a lower accuracy than the multinomial model
# 72%

pred_probs = clf.predict_proba(test_feat_repeat)

import matplotlib.pyplot as plt
plt.hist(np.max(pred_probs, axis=1), bins=10, color='blue', edgecolor='black')
plt.show()

# Next thing to do is to try and see if there is a benefit to using the multilabel model
# I think the thing to check would be to, after fitting the multi-label model look at the
# observations that have a predicted probability larger than lets say 0.8 for each class.
# And then measure the accuracy of the multinomial model on these observations.

# or look at observations with and without multiple labels in the test set seprarately



# Multi-label
# Now looking at xgboost accuracy:
clf = xgb.XGBClassifier(tree_method="hist")
clf.fit(train_features_py, train_res_py)
np.testing.assert_allclose(clf.predict(test_features_py), test_res_py)
accuracy_score(test_res_py, clf.predict(test_features_py))

# Overall Accuracy of R model:
predict_prob.shape
# changing to class predictions matrix, using largest probabulity
max_indices = np.argmax(predict_prob, axis=1)
predict_class = np.zeros_like(predict_prob)
predict_class[np.arange(predict_class.shape[0]), max_indices] = 1

accuracy_score(predict_class, test_res_py)

t_array = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 1]
])
p_array = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0]
])

accuracy_score(t_array, p_array)

# For each class (Task View) want to get observations that have a predicted probability larger than 0.8











