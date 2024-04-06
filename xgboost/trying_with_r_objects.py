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

# Checking which rownumber belongs to GLMMadaptive package.
# I want to check the outputted predicted probabilities
row_names = robjects.r['rownames'](test_features_py)
col_names = robjects.r['colnames'](test_res_py)
col_names = list(col_names)
row_names = list(row_names)
row_number = row_names.index("GLMMadaptive")
row_names = robjects.r['rownames'](train_features_py)
row_number = row_names.index("lme4")


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



#-------------------------------------------------------------#

# Multi-label
# Now looking at xgboost accuracy:
clf = xgb.XGBClassifier(tree_method="hist")
clf.fit(train_features_py, train_res_py)
np.testing.assert_allclose(clf.predict(test_features_py), test_res_py)
accuracy_score(test_res_py, clf.predict(test_features_py))

# Predicted probabilities
predicted_probabilities = clf.predict_proba(test_features_py.T)
predicted_probabilities = clf.predict_proba(train_features_py)
# After seeing that lme4 had a low pred probability in the multinomial model,
# I wanted to see what prob it would have in the multi-label model. 
# In the multi-label model, the predicted probabilities for the classes that lme4
# belongs to are large. This is good but not suprising as the package was in the training set,
# with multiple labels.
# But this does reinforce the idea that the motivation for why I want to try multi-label model is so that
# recommendation probabilities for packages that should belong to multiple classes are not
# compromised.
np.round(predicted_probabilities[1569], 2)
np.where(np.round(predicted_probabilities[1569], 2) > 0.9)[0]

# Overall Accuracy of R model:
predict_prob.shape
# changing to class predictions matrix, using largest probabulity
max_indices = np.argmax(predict_prob, axis=1)
predict_class = np.zeros_like(predict_prob)
predict_class[np.arange(predict_class.shape[0]), max_indices] = 1

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

# accuracy_score looks at proportion of rows that are predicted correctly
# in example 2 out of 3 rows are predicted correctly, hence 0.666
accuracy_score(t_array, p_array)
accuracy_score([[1, 0, 0]], [[1, 0, 1]])
accuracy_score([1, 0, 0], [1, 0, 1])

multi_label_rows_test = np.zeros(test_res_py.shape[0])
for i in range(0, test_res_py.shape[0]):
    if sum(test_res_py[i,]) > 1:
        multi_label_rows_test[i] = 1


accuracy_score(predict_class, test_res_py)
accuracy_score(predict_class[2:3,], test_res_py[2:3,])

#  from xgboost model
predict_prob_xg = clf.predict_proba(test_features_py.T)
predict_prob_xg[2,]
predict_class_xg = np.where(predict_prob_xg > 0.6, 1, 0)
accuracy_score(predict_class_xg[2:3,], test_res_py[2:3,])

# for each row sum the number of labels
sum(np.sum(predict_class_xg, axis=1) > 1)
sum(np.sum(test_res_py, axis=1) > 1)
# accuracy_score for multi-label model
# seems to have a lower overall accuracy
accuracy_score(predict_class_xg, test_res_py)
# lets see how well it does at the multi-label observations
accuracy_score(predict_class_xg[np.where(multi_label_rows_test == 1)],test_res_py[np.where(multi_label_rows_test == 1)])
accuracy_score(predict_class[np.where(multi_label_rows_test == 1)],test_res_py[np.where(multi_label_rows_test == 1)])

# In practice as we want a recommendation system, it makes more sense to look at accuracy of top 20  or so
# recommendations for each class,
# For each class (Task View) want to get observations that have a predicted probability larger than 0.8

# for each coloum of predict_prob_xg get row numbers for largest 50 values
top_50_row_ind = np.argsort(predict_prob_xg, axis=0)[::-1][0:20]

# use top_50_row_ind to exract values from test_res_py 
selected_values = np.zeros_like(top_50_row_ind)
for i in range(0, test_res_py.shape[1]):
    selected_values[:,i] = test_res_py[top_50_row_ind[:,i],i]

# take mean of an array
np.mean(selected_values)


# for each coloum of predict_prob_xg get row numbers for largest 50 values
top_50_row_ind = np.argsort(predict_prob, axis=0)[::-1][0:20]

# use top_50_row_ind to exract values from test_res_py 
selected_values = np.zeros_like(top_50_row_ind)
for i in range(0, test_res_py.shape[1]):
    selected_values[:,i] = test_res_py[top_50_row_ind[:,i],i]

# take mean of an array
np.mean(selected_values)





