
#' Helper function that returns or saves objects created by functions in this package.
#'
#' @param list_to_return
#'
#' @return
#'
#' @examples
save_or_return_objects = function(list_to_return, limiting_n_observations, save_output, save_path, file_name){

  if(attributes(list_to_return)$TEST){
    attr(list_to_return, "limiting_n_observations") = limiting_n_observations
  }

  # If save_get_NLP set to TRUE then the object is saved to get_NLP_save_path
  # The default path is in the test directory, as I want to save objects so that they
  # do not have to be recreated every time in a test.
  if(save_output){

    saveRDS(list_to_return, file = file.path(save_path, file_name))
    message("Objects:", paste(names(list_to_return), collapse = ", "), " have been saved to the path: ~/", file.path(save_path))

  }else{

    return(list_to_return)

  }

}


