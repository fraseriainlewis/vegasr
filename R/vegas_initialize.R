#' Initialize vegas
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



