# File to experiment to try and get model to converge
############################################################################################################

get_input_stored = TRUE
get_input_path = "tests/testthat/fixtures/get_CRAN_logs_output/get_CRAN_logs_output.rds"



if(get_input_stored){
  get_CRAN_logs_output = readRDS(get_input_path)

}else{

  # Get required objects from CTVsuggest:::get_CRAN_logs()
  get_CRAN_logs_output = CTVsuggestTrain:::get_CRAN_logs(TEST = TEST, limiting_n_observations = limiting_n_observations)

}

#### ----------------------------------------------------------------------------------------------- ####




#### ----------------------------------------------------------------------------------------------- ####

# Loading vector of packages with no Task View assignment that do not meet threshold
no_tsk_pckgs_meet_threshold = base::intersect(get_CRAN_logs_output$no_tsk_pckgs_meet_threshold, get_CRAN_logs_output$final_package_names)
no_tsk_pckgs_meet_threshold = unique(no_tsk_pckgs_meet_threshold)

# The training and testing sets are made up of the packages that either have:
# no Task View but meet download threshold
# (response_matrix[no_tsk_pckgs_meet_threshold,])
# or has an assigned Task View
# (response_matrix[response_matrix[,"none"] == 0,])

# combining the two sets
labelled_data_res = (rbind(get_CRAN_logs_output$response_matrix[get_CRAN_logs_output$response_matrix[,"none"] == 0,],    get_CRAN_logs_output$response_matrix[no_tsk_pckgs_meet_threshold,]))
labelled_data_features = get_CRAN_logs_output$features[rownames(labelled_data_res),]


# labelled_data_res_df = rbind(response_df[!(response_df[,"TaskViews"] == "none"),], response_df[no_tsk_pckgs_meet_threshold,])

set.seed(3)
split1<- sample(c(rep(0, 0.8 * nrow(labelled_data_res)), rep(1, 0.2 * nrow(labelled_data_res))))
table(split1)
train_res = labelled_data_res[split1 == 0,]
train_features = labelled_data_features[split1 == 0,]
test_res = labelled_data_res[split1 == 1,]
#test_res_df = labelled_data_res_df[split1 == 1,]
test_feature = labelled_data_features[split1 == 1,]

#### ----------------------------------------------------------------------------------------------- ####


#### ----------------------------------------------------------------------------------------------- ####
##### LASSO #####

# removing row that has missing features
# train_res = train_res[!apply(as.matrix(train_features),1, function(x){any(is.na(x))}),]
# train_features = train_features[!apply(as.matrix(train_features),1, function(x){any(is.na(x))}),]

train_res = as.matrix(train_res)
train_features = as.matrix(train_features)



train_sparse <- Matrix::sparse.model.matrix(~., as.data.frame(train_features))
train_res_sparse <- Matrix::sparse.model.matrix(~0 + ., as.data.frame(train_res))

message("Training model")
set.seed(3)
model_multinom_cv = glmnet::cv.glmnet(x = train_sparse[,1:20],  y = train_res, family = "multinomial", alpha = 1, trace.it = 1)
model_multinom_cv = glmnet::cv.glmnet(x = train_sparse,  y = train_res, family = "multinomial", alpha = 1, trace.it = 1)
#### ----------------------------------------------------------------------------------------------- ####



#### ----------------------------------------------------------------------------------------------- ####
#### Accuracy ####

model = model_multinom_cv
test_feature = test_feature[,1:20]
predict_class = predict(model, newx = cbind(rep(1, nrow(test_feature)),as.matrix(test_feature)), s = "lambda.min",  type = "class")
# Getting accuracy of model after applying lasso with min Lambda
predict_class = factor(predict_class[,1], levels = c(RWsearch::tvdb_vec(get_CRAN_logs_output$tvdb), "none"))



model_accuracy = mean(test_res[cbind(1:nrow(test_res), predict_class)], na.rm = T)
model_accuracy = 100*model_accuracy

# apply(test_res[which(test_res[cbind(1:nrow(test_res), predict_class)] == 0),],2,sum)
#### ----------------------------------------------------------------------------------------------- ####

