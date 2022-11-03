# Creates objects for fixtures folder, which are then used for testing.
# Should run with working directory set to package directory.

# get_data_output
get_data(save_output = TRUE, file_name = "get_data_output.rds")
get_data(TEST = TRUE, save_output = TRUE, file_name = "get_data_output_test.rds")
# get_NLP_output
get_NLP(get_input_stored = TRUE, save_output = TRUE)
