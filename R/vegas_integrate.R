#' Multidimensional Integration using Vegas
#'
#' @param f An R function that takes a numeric vector and returns a numeric value.
#' @param lower A vector of lower integration limits for each dimension, e.g. c(-1.,-1.-1)
#' @param upper A vector of upper integration limits for each dimension, e.g. c(1.,1.1)
#' @param nitn Number of iterations.
#' @param neval Number of function evaluations per iteration.
#' @param ... Additional arguments passed to the vegas integrator.
#' @importFrom glue glue
#' @examples
#' library(mvtnorm)
#'
#' myf<-function(x){
#'  #x <- as.matrix(x) # necessary to avoid R auto-flattening arrays
#'  res<-dmvnorm(x, mean =
#'                 matrix(c(0.5, -0.2, 0.1),nrow=1),
#'               sigma = matrix(data=c(
#'                 1.0, 0.5, 0.2,
#'                 0.5, 1.2, 0.3,
#'                 0.2, 0.3, 0.8),ncol=3,byrow=FALSE))
#'  return(res)
#'  }
#'
#' vegas_integrate(f=myf,lower=c(-5.,-5.,-5.),upper=c(5.,5.,5.))
#'
#' @export
vegas_integrate <- function(f, lower,upper, nitn = 10, neval = 1000, ...) {

  if (is.null(getOption("vegas_initialized"))) {
    vegas_initialize()
  }

  # conversion. Important because R uses fortran col-order
  # scoping matters - r_to_py() has auto conversion but does not scope inside function calls
  # so import_main is needed, but this only maps references and so conversion needed, e.g. py_func below
  # whereas python uses C row-order. e.g. so in the example
  # #x <- as.matrix(x) needs to be added unless we use py_func() auto-convern
  #
  main <- reticulate::import_main(convert = FALSE)
  main$r_func <- reticulate::py_func(f)

  main$Rlower<-as.numeric(lower)
  main$Rupper<-as.numeric(upper)

  stringpart1<-r"(
  #### now use R function
  )"

str1<-r"(r_func(x))"

stringpart<-glue::glue('
# Integration limits
#lower = np.array([-5.0, -5.0, -5.0])
#upper = np.array([5.0, 5.0, 5.0])
lower = np.array(Rlower,dtype=np.float64)
upper = np.array(Rupper,dtype=np.float64)


## this decorator is using (BATCH,dim) - C-row-order
@vegas.lbatchintegrand
def ff(x):
  #print(x.shape)
  return {str1}

# Initialize the integrator
integ2 = vegas.Integrator([[l, u] for l, u in zip(lower, upper)])
# Adaptation phase
integ2(ff, nitn=10, neval=1000)
# Final integration
result2 = integ2(ff, nitn=10, neval=1000)
#print(result2.summary())
',.trim=FALSE)

bigstring<-paste(stringpart,sep="")
  #return(bigstring)
reticulate::py_run_string(bigstring)
main <- reticulate::import_main(convert = FALSE)
cat(reticulate::py_to_r(main$result2$summary()))

#return(results)

}


