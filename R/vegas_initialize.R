#' @title Initialize vegas
#'
#' @description Initialize vegas by checking python is available, that the vegas python package is installed,
#' load the python class needed by vegasr, and set an environment variable.
#' It will prompt for \code{\link{vegas_install}} if needed.
#'
#' This function should be run after library(vegas) the first time this is called in an R session. See examples in ?vegas
#' @export
vegas_initialize <- function() {

  if (!is.null(options("vegas_initialized")$vegas_initialized)) {
    message("vegas is already initialized")
    return(NULL)
  }

  if (!nchar(Sys.getenv("RETICULATE_PYTHON")) > 0) {
    if (!reticulate::virtualenv_exists(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr")))
      stop("No vegas environment found. Use `vegas_install()` to get started.")

    if (!"vegas" %in% reticulate::py_list_packages(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr"))$package)
      stop("vegas was not found in your environment. Use `vegas_install()`",
           "to get started.")

    reticulate::use_virtualenv(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr"))
  }

  vegasr_pyexec(pyfile = system.file("python", "vegasr_class.py",
                                     package = "vegasr"))

  vegasr_pyexec(pyfile = system.file("python", "initialize_vegasPython.py",
                                     package = "vegasr"))

  vegas_version <- vegasr_pyget("vegas_version")
  message("successfully initialized vegas version: ", vegas_version)
  options("vegas_initialized" = TRUE)

}



