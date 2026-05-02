#' Install vegas in conda or virtualenv environment
#'
#' @description Install vegas in a self-contained environment
#' @param ask logical; ask whether to proceed during the installation. By
#'   default, questions are only asked in interactive sessions.
#' @param force ignore if vegas is already present and install
#'   it anyway.
#'
#' @details The function checks whether a suitable installation of Python is
#'   present on the system and installs one via
#'   [reticulate::install_python()] otherwise. It then creates a
#'   virtual environment with the necessary packages in the default location
#'   chosen by [reticulate::virtualenv_root()].
#'
#'   If you want to install a different version of Python than the default, you
#'   should call [reticulate::install_python()] directly. If you want
#'   to create or use a different virtual environment, you can use, e.g.,
#'   `Sys.setenv(VEGAS_PYTHON = "path/to/directory")`.
#'
#'
#' @examples
#' \dontrun{
#' # install the latest version of spaCy
#' vegas_install()
#'
#' vegas_install(force = TRUE)
#'
#' # install vegas to an existing virtual environment
#' Sys.setenv(RETICULATE_PYTHON = "path/to/python")
#' vegas_install()
#' }
#'
#' @export
vegas_install <- function(ask = interactive(),
                          force = FALSE) {

  if (nchar(Sys.getenv("RETICULATE_PYTHON")) > 0) {
    message("You provided a custom RETICULATE_PYTHON, so we assume you know what you ",
            "are doing managing your virtual environments. Good luck!")
  } else if (!reticulate::virtualenv_exists(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr"))) {
    # this has turned out to be the easiest way to test if a suitable Python
    # version is present. All other methods load Python, which creates
    # some headache.
    t <- try(reticulate::virtualenv_create(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr")), silent = TRUE)
    if (methods::is(t, "try-error")) {
      permission <- TRUE
      if (ask) {
        permission <- utils::askYesNo(paste0(
          "No suitable Python installation was found on your system. ",
          "Do you want to run `reticulate::install_python()` to install it?"
        ))
      }

      if (permission) {
        if (utils::packageVersion("reticulate") < "1.19")
          stop("Your version or reticulate is too old for this action. Please update")
        python <- reticulate::install_python()
        reticulate::virtualenv_create(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr"),
                                      python = python)
      } else {
        stop("Aborted by user")
      }
    }
    reticulate::use_virtualenv(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr"))
  } else {
    reticulate::use_virtualenv(Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr"))
  }

  vegas_pkg <- "vegas"

  if (py_check_installed(vegas_pkg) & !force) {
    warning("Skipping installation. Use `force` to force installation or update.")
    return(invisible(NULL))
  }

  reticulate::py_install(vegas_pkg, envname = Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr"))

  message("Installation of spaCy version ",
          py_check_version("vegas", envname = Sys.getenv("VEGAS_PYTHON", unset = "r-vegasr")),
          " complete.")

  invisible(NULL)
}


