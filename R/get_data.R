#' Extracts data used to create features for model
#'
#' @details
#'    The `get_data()` function is run inside [get_NLP()].
#'
#' `get_data()` extracts the following types of data:
#'    * Task View data, using the [download_taskview_data()].
#'    * CRAN data from the CRAN package repository using [tools::CRAN_package_db()].
#'
#' `get_data()` also runs the [cranly::clean_CRAN_db()] function on the CRAN data repository.
#'
#'
#' @param TEST logical. Default is [`FALSE`]. If [`TRUE`], then a subset of the data that is extracted from CRAN is selected. This is to speed up testing.
#'
#' More precisely, if [`TRUE`] a random selection of rows from `CRAN_data` is selected, where the number of rows
#' chosen is given by `limiting_n_observations`.
#'
#' @param limiting_n_observations Integer that decides the size of the subset of `CRAN_data`, when `TEST` is [`TRUE`].
#'
#'
#' @param save_output logical. Default is [`FALSE`]. If [`TRUE`], then the list that is returned is saved to the path set by
#'    `save_path`.
#' @param save_path string. Sets the path where the list created by the function will be saved,
#'    which is when `save_output` is set to [`TRUE`]
#' @param file_name string. Sets the file name for the saved object.
#'
#'
#' @returns `get_data` returns data objects required for rest of scripts involved in training the model:
#'\itemize{
#'   \item CRAN_data - Data extracted from CRAN package repository using [tools::CRAN_package_db()]. Duplicated packages removed. If `TEST` = [`TRUE`] then a random selection of rows `CRAN_data` of length `limiting_n_observations` is selected.
#'   \item all_CRAN_pks - Package names that have data included in the `CRAN_data` object.
#'   \item CRAN_cranly_data - [`data.frame`] with class [`cranly_db`] that is created using [`cranly::clean_CRAN_db()`]. The [`data.frame`] has the same variables as `CRAN_data`.
#'   \item tvdb - list object of class `ctvlist` that contains information about the Task Views. This is downloaded using the function `CTVsuggest:::download_taskview_data()` which is a modified version of [`RWsearch::tvdb_down()`]
#'   \item TEST - returns the `TEST` value used in the function. As this function is used within the `get_nlp` function, and information about whether a subset of the full data
#'   is being used needs to be carried forward.
#' }
#'
#' @examples
#' \donttest{
#'    CTVsuggest:::get_data(TEST = TRUE, limiting_n_observations = 100)
#'    }
get_data = function(TEST = FALSE,
                    limiting_n_observations = 100,
                    save_output = FALSE, save_path = "tests/testthat/fixtures/get_data_output", file_name){


  message("Downloading package metadata from CRAN package repository")

  tvdb = CTVsuggestTrain:::download_taskview_data()

  # Data extracted from CRAN package repository
  CRAN_data = tools::CRAN_package_db()

  # There are some packages that are given twice.
  # Most common difference in rows labeled as belonging to the same package is dependency on a more recent version of R.
  # I have ignored the extra information and just removed duplicated packages.
  CRAN_data = CRAN_data[!duplicated(CRAN_data$Package),]


  ############ TESTING ###########
  # Limits number of observations in dataset to speed up tests
  if(TEST){
    test_sample = sample(size = limiting_n_observations, x = nrow(CRAN_data))
    CRAN_data = CRAN_data[test_sample,]
  }
  ################################

  # all_CRAN_pks is all of the current packages available in CRAN. Or a subset of these if `TEST` is set to `TRUE`.
  all_CRAN_pks = CRAN_data$Package


  # CRAN_data cleaned and converted into form that can be used by cranly
  CRAN_cranly_data = cranly::clean_CRAN_db(packages_db = CRAN_data)

  ############ Creating and Returning FINAL object ############
  # Creating object to be returned. Which is a list made up of objects needed upstream
  list_to_return = list("CRAN_data" = CRAN_data, "all_CRAN_pks" = all_CRAN_pks, "CRAN_cranly_data" = CRAN_cranly_data, "tvdb" = tvdb, "TEST" = TEST)

  CTVsuggestTrain:::save_or_return_objects(TEST = TEST, list_to_return = list_to_return, limiting_n_observations = limiting_n_observations,
                                           save_output = save_output, save_path = save_path, file_name = file_name)

}

