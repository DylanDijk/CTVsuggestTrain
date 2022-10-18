
#' Title
#'
#' @param TEST
#' @param limiting_n_observations
#'
#' @return Data objects required for rest of scripts involved in training the model
#'
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

  ### all_CRAN_pks is all of the current packages available in CRAN
  all_CRAN_pks = CRAN_data$Package


  ## CRAN_data cleaned and converted into form that can be used by cranly
  CRAN_cranly_data = cranly::clean_CRAN_db(packages_db = CRAN_data)


  return(list("CRAN_data" = CRAN_data, "all_CRAN_pks" = all_CRAN_pks, "CRAN_cranly_data" = CRAN_cranly_data, "tvdb" = tvdb, "TEST" = TEST))

}

