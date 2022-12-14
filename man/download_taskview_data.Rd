% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/download_taskview_data.R
\name{download_taskview_data}
\alias{download_taskview_data}
\title{Downloads Task View data}
\usage{
download_taskview_data(repos = getOption("repos")[1])
}
\arguments{
\item{repos}{Default option set to the CRAN repository "https://cran.rstudio.com/".}
}
\value{
Data of the current Task Views available.
}
\description{
This is a modification of the RWsearch::tvdb_down function, this modified version does not save a file to disk.
Default option is to extract Task View data from CRAN Task View repository: "https://cran.rstudio.com/src/contrib/Views.rds"
}
