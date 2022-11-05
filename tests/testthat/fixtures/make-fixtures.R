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


