
#' Gets data for model training. This function is run inside `CTVsuggest:::get_NLP()`.
#'
#' @param TEST logical. If TRUE, then a subset of the data that is extracted from CRAN is selected. This is to speed up testing.
#'
#' More precisely, if TRUE a random selection of rows `CRAN_data` of length `limiting_n_observations` is selected.  Default is [`FALSE`].
#' @param limiting_n_observations Integer that decides the size of the subset of `CRAN_data`, when `TEST` is [`TRUE`].
#'
#' @return Data objects required for rest of scripts involved in training the model
#'\itemize{
#'   \item CRAN_data - Data extracted from CRAN package repository using [tools::CRAN_package_db()]. With duplicated packages removed. If TRUE then a random selection of rows `CRAN_data` of length `limiting_n_observations` is selected.
#'   \item all_CRAN_pks - Package names that have data included in the `CRAN_data` object.
#'   \item CRAN_cranly_data - cranly::clean_CRAN_db a [`data.frame`] with the same variables as CRAN_data
#'   \item tvdb - list object of class `ctvlist` that contains information about the Task Views. This downloaded using the function `CTVsuggest:::download_taskview_data()` which is a modified version of [`RWsearch::tvdb_down()`]
#'   \item TEST - returns the `TEST` value used in the function. As the function is used within the `get_nlp` function.
#' }
#'
#' @examples
get_data = function(TEST = FALSE, limiting_n_observations = 100){

  message("Downloading package metadat from CRAN package repository")

  tvdb = CTVsuggest:::download_taskview_data()

  # CRAN snapshot
  ## Data extracted from CRAN package repository
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

  ### all_CRAN_pks is all of the current packages available in CRAN. Or subset of these if `TEST` is set to `TRUE`
  all_CRAN_pks = CRAN_data$Package


  ## CRAN_data cleaned and converted into form that can be used by cranly
  CRAN_cranly_data = cranly::clean_CRAN_db(packages_db = CRAN_data)


  return(list("CRAN_data" = CRAN_data, "all_CRAN_pks" = all_CRAN_pks, "CRAN_cranly_data" = CRAN_cranly_data, "tvdb" = tvdb, "TEST" = TEST))

}

