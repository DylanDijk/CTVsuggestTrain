# Creates objects for fixtures folder, which are then used for testing.
# Should run with working directory set to package directory.

# get_data_output
get_data(save_output = TRUE, file_name = "get_data_output.rds")
## get_data_output TEST
get_data(TEST = TRUE, save_output = TRUE, file_name = "get_data_output_test.rds")


# get_NLP_output
get_NLP(get_input_stored = TRUE, save_output = TRUE)
## get_NLP_output TEST
get_NLP(get_input_stored = TRUE, save_output = TRUE,
        get_input_path = "tests/testthat/fixtures/get_data_output/get_data_output_test.rds",
        save_path = "tests/testthat/fixtures/get_NLP_output", file_name = "get_NLP_output_test.rds")


# get_create_features
get_create_features(get_input_stored = TRUE, save_output = TRUE)
## get_create_features TEST
get_create_features(get_input_stored = TRUE, save_output = TRUE,
                    get_input_path = "tests/testthat/fixtures/get_NLP_output/get_NLP_output_test.rds",
                    save_path = "tests/testthat/fixtures/get_create_features_output", file_name = "get_create_features_output_test.rds")


# get_CRAN_logs
get_CRAN_logs(get_input_stored = TRUE, save_output = TRUE)
## get_CRAN_logs TEST
get_CRAN_logs(get_input_stored = TRUE, save_output = TRUE,
              get_input_path = "tests/testthat/fixtures/get_create_features_output/get_create_features_output_test.rds",
              save_path = "tests/testthat/fixtures/get_CRAN_logs_output", file_name = "get_CRAN_logs_output_test.rds")



# To train the model using the fixtures we have created and saved here, we run:
  # This saves the model object and its accuracy in the OUTPUT folder
Train_model(get_input_stored = TRUE, save = TRUE)
