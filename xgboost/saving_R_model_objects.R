save(test_feature, file = "xgboost/test_feature.RData")
save(test_res, file = "xgboost/test_res.RData")
save(train_features, file = "xgboost/train_features.RData")
save(train_res, file = "xgboost/train_res.RData")

predict_prob = predict(model, newx = cbind(rep(1, nrow(test_feature)),as.matrix(test_feature)), s = "lambda.min",  type = "response")
predict_prob = predict_prob[,,1]
save(predict_prob, file = "xgboost/predict_prob.Rdata")
