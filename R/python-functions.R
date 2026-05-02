vegasr_pyassign <- function(pyvarname, values) {
    #main <- reticulate::import_main()
    #eval(parse(text = sprintf("main$%s <- reticulate::r_to_py(values)", pyvarname)))
  main <- reticulate::import_main(convert = FALSE)
  #main[[pyvarname]] <- reticulate::np_array(reticulate::r_to_py(values))
  main[[pyvarname]] <- reticulate::np_array(values)
  }

vegasr_pyget <- function(pyvarname) {
    main <- reticulate::import_main()
    return(eval(parse(text = sprintf("main$%s", pyvarname))))
}

vegasr_pyexec <- function(pystring = NULL, pyfile = NULL) {
    if (!is.null(pystring)) {
        reticulate::py_run_string(pystring)
    }
    if (!is.null(pyfile)) {
        reticulate::py_run_file(pyfile)
    }
}
