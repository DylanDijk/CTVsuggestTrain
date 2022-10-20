# Function downloads Task View data
# This is a modification of the RWsearch::tvdb_down function, this modified version does not save a file to disk

#' Title
#'
#' @param repos
#'
#' @return Data of the current Task Views available
#'
#'
#' @examples
#' \donttest{
#' tvdb = download_taskview_data()
#' }
download_taskview_data = function(repos = getOption("repos")[1]){
  urlrds <- paste0(repos, "/src/contrib/Views.rds")
  dest <- tempfile()
  trdl <- RWsearch:::trydownloadurl(urlrds, dest)
  if (trdl != 0) {
    message(paste("File does not exist:", urlrds))
    message("Is your repository out of service? Check with cranmirrors_down().")
    return(invisible(NULL))
  }
  tvdb <- readRDS(dest)
  names(tvdb) <- ntv <- sapply(tvdb, function(x) x$name)
  tvdb <- tvdb[sort(ntv)]
  return(tvdb)
}
