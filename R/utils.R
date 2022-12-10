
#' Helper function that returns or saves objects created by functions in this package.
#'
#' @param TEST Logical. If TRUE then the objects it is saving have been generated from a test.
#' @param list_to_return List. List of objects that you want to be saved.
#' @param limiting_n_observations Integer. If objects are generated from a TEST then, `limiting_n_observations`.
#' @param save_output Logical. Determines whether `list_to_return` is saved to disk or return into the current R environment.
#' @param save_path String. Path where `list_to_return` will be saved.
#' @param file_name String. File name of saved object.
#'
#' @return The function saves or returns the `list_to_return` list, depending on value of `save_output`.
#'
save_or_return_objects = function(TEST, list_to_return, limiting_n_observations, save_output, save_path, file_name){

  # Assigning attributes to object that will be returned
  attr(list_to_return, "date") = Sys.Date()
  attr(list_to_return, "TEST") = TEST


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


