#' Multidimensional Integration using Vegas+
#'
#' @description An R wrapper for the Python numerical integration library vegas. This allows integrals defined as R functions to be passed to the python library for computation. Some minimal requirements apply to the functions. See Details.
#'
#' @details
#' The \href{https://pypi.org/project/vegas/}{vegas} Python library implements the 2020 VEGAS+ adaptive Monte Carlo numerical integration algorithm.
#' This is update of the original vegas algorithm from 1978 and by the original author G. P. Lepage, who also maintains the Python library. This library is a thin R wrapper making the key functions accessible from R.
#' For more information on the Vegas algorithm, visit \url{https://vegas.readthedocs.io/} and the VEGAS+ arXiv article is \href{https://arxiv.org/abs/2009.05112}{here}.
#'
#' The R function passed defines an integral and its structure and arguments need to meet some minimum requirements:
#'
#' \itemize{
#'   \item The first argument passed to the R function from Python vegas will be a numerical matrix of shape \code{[BATCH,Dim]}, and vegas will expect back a numerical vector of length Dim. Each row in the matrix is a single set of values of the integration variables, e.g. for a 5-D integrand then this matrix must have 5 columns. Reticulate takes care of conversion between R and Python matrices and vectors but the dimensions must be as described.
#'   \item Additional named arguments can be passed and these should always be matrix() objects, with integers converted to float (or have trailing period added to avoi ambiguity)
#'   \item  If an additional argument is a scalar, say svalue, then use mvalue<-matrix(data=c(svalue,nrow=1), coercing to float first if necessary, and in the R function use \code{mvalue[1]}.
#' }
#'
#' @references
#' G. P. Lepage, “A New Algorithm for Adaptive Multidimensional Integration” J. Comp. Phys. 27, 192–203 (1978).
#'
#' G. P. Lepage, “Adaptive multidimensional integration: vegas enhanced” J. Comp. Phys. 439 (2021) 110386.
#'
#' \url{https://doi.org/10.1016/j.jcp.2021.110386}
#'
#' @param f An R function that takes a matrix and returns a vector. See details and examples.
#' @param lower A vector of lower integration limits for each dimension, e.g. c(-1.,-1.-1)
#' @param upper A vector of upper integration limits for each dimension, e.g. c(1.,1.1)
#' @param nitn_warm Number of iterations for Vegas warmup
#' @param neval_warm Number of function evaluations per iteration in warmup
#' @param nitn Number of iterations post-warmup.
#' @param neval Number of function evaluations per iteration post-warmup.
#' @param errTol  the % error target, default is 1, i.e. error is 1% of current estimated integral value
#' @param maxIter max number of iteration blocks to run to achieve errTol. Each block comprises nitn iterations
#' @param seed random number seed for vegas sampling generating. set for reproducible results.
#' @param extra_args a named list of additional arguments passed to the function f.
#' These must be numeric vectors or matrices. See details.
#' @importFrom glue glue
#' @examples
#' \dontrun{
#' # Use library(mvtnorm) first for 3-D Multivariate Normal Density
#' library(mvtnorm)
#' mu<-matrix(c(0.5, -0.2, 0.1),nrow=1) # note matrix
#' cov<-matrix(data=c(
#'  1.0, 0.5, 0.2,
#'  0.5, 1.2, 0.3,
#'  0.2, 0.3, 0.8),ncol=3,byrow=FALSE)
#'
#' # use built-in pmvnorm()
#' lower=c(-0.5,-0.5,-0.5); upper=c(2.,1.,3.)
#' res_builtin<-pmvnorm(lower=lower,upper=upper,
#'                     mean=as.numeric(mu), # coercion as needs vector
#'                     sigma=cov)
#' print(res_builtin)
#'
#' ## now use vegas to compute same integral
#' library(vegasr)
#' ## Important - run next line in each R session to ensure python is ready
#' vegas_initialize()
#'
#' # the integrand function MUST take a matrix of dimension [BATCH,M] as first
#' # argument and return a vector of length M (or 1-row matrix).
#' # Any number of other additional *named* arguments are allowed and should all
#' # be matrices. See ?vegas_integrate for more details.
#' myf<-function(x,mu,cov){
#'   res<-dmvnorm(x,
#'                mean = mu, # this is a 1-row matrix, dmvnorm accepts this
#'                sigma=cov)
#'   return(res)
#' }
#'
#' ## See help page for descriptions of warm and nitn and neval.
#' vegas_result<-vegas(f=myf,
#'                               lower=lower, upper=upper,
#'                               nitn_warm = 10, neval_warm = 10000,
#'                               nitn = 10, neval = 10000,
#'                               errTol=0.1, maxIter=20,seed=99999,
#'                               extra_args=list(mu=mu, cov=cov))
#' # extra_args are additional arguments needed for myf
#' print(vegas_result)
#'
#' }
#'
#'
#'
#' @export
vegas <- function(f, lower,upper, nitn_warm = 10, neval_warm = 1000,
                                            nitn = 10, neval = 1000, errTol=1,maxIter=5,
                                            seed = 99999,extra_args = list()) {

  if (is.null(getOption("vegas_initialized"))) {
    vegas_initialize()
  }

  run_checks() # empty - to be completed - named argument checks etc

  # conversion. Important because R uses Fortran col-order
  # scoping matters - r_to_py() has auto conversion but does not scope inside function calls
  # so import_main is needed, but this only maps references and so conversion needed, e.g. py_func below
  # whereas python uses C row-order. e.g. so in the example
  # #x <- as.matrix(x) needs to be added unless we use py_func() auto-convern
  #
  main <- reticulate::import_main(convert = FALSE)
  main$r_func <- reticulate::py_func(f)


  main$Rlower<-as.numeric(lower)
  main$Rupper<-as.numeric(upper)

  #main$cleanstart<-as.logical((cleanstart[1]))

  main$nitn_warm<-as.integer((nitn_warm[1]))
  main$neval_warm<-as.integer((neval_warm[1]))
  main$nitn<-as.integer((nitn[1]))
  main$neval<-as.integer((neval[1]))

  main$RmaxIter<-as.integer((maxIter[1]))
  main$RerrTol<-as.numeric((errTol[1]))

  main$Rseed<-as.integer((seed[1]))

  # 1. Initialize local variables
  noms <- character(0)
  args <- extra_args
  #args <- list(...)

  if (length(args) > 0){
    #cat("parsing additional arguments\n")
    m<-length(args);
    #args<-list(...)
    noms<-names(args)
    for(i in 1:m){
      #print(args[[i]])
      vegasr_pyassign(noms[i],args[[i]])

    }
    #cat("names of extra args=\n");print(noms);cat("\n")
  }


# 1. find ou
str1<-paste(paste(paste(noms,"=",sep=""),noms,sep=""),collapse=",") # y=y,z=z
str2<-paste(c("x",noms),collapse=",") # x,y,z
str2b<-paste(c("x_copy",noms),collapse=",") # x_copy,y,z
str3<-paste(c(noms),collapse=",") # y,z
str4<-paste(paste("self",noms,sep="."),collapse=",") # self.y, self.z
str5<-paste(paste(paste("        self",noms,sep="."),"=",noms),collapse="\n") # self.y=y\nself.z=y
#  stringpart1<-r"(
#  #### now use R function
#  )"

#str1<-r"(r_func(theta,self.y,self.z))"
#str2<-r"(r_func(x,y,z))"

stringpart<-glue::glue('

## this decorator is using (BATCH,dim) - C-row-order
@vegas.lbatchintegrand
def r_func_lbatch({str2}):
  #print(x.shape)
  #
  #print("here")
  #return r_func({str2})
  #return r_func(x.copy(),mu.copy(),cov.copy())
  #return r_func(x.tolist(), mu.tolist(), cov.tolist())
  x_copy = np.array(x, dtype=np.float64, copy=True)
  res=np.array(r_func({str2b})) #(x_copy, mu, cov))
  #print(f"shape={{res.shape}} and object itself is={{res}}")
  return res


@vegas.lbatchintegrand
class vegasHelper:
    def __init__(self,{str3}):
{str5}

    def __call__(self, theta):
        return(r_func_lbatch(theta,{str4}))

#newf = vegasHelper(y=y,z=z) # str1
newf = vegasHelper({str1})

# Integration limits
lower = np.array(Rlower,dtype=np.float64)
upper = np.array(Rupper,dtype=np.float64)

# Start from clean slate and run warm-up
vegas_obj.clear_results()
#explicitly set seed
gvar.ranseed(Rseed)
integ2 = vegas.Integrator([[l, u] for l, u in zip(lower, upper)])
# Adaptation phase # no results stored
integ2(newf, nitn=nitn_warm, neval=neval_warm)

# now decide how many update blocks to run, if errTol=NULL
# then run maxIter blocks, if errTol is not null then run until error hit
i=0
iMax=RmaxIter
vegas_obj.success=True
while (i==0) or (ests[1]>((RerrTol/100)*ests[0]) and i<iMax):
    # integration storing interative results, still potentially adapting grid
    result2 = integ2(newf, nitn=nitn, neval=neval)
    vegas_obj.add_results(result2) # save into object
    ests=vegas_obj.get_final_wt_results() # get the current overall estimate and error

    print(i)
    #print(ests)
    #print(result2.summary())
    i+=1

if ests[1]>((RerrTol/100)*ests[0]):
    vegas_obj.success=False

',.trim=FALSE)

bigstring<-paste(stringpart,sep="")
#return(bigstring)
reticulate::py_run_string(bigstring)
main <- reticulate::import_main(convert = FALSE)
#cat(reticulate::py_to_r(main$result2$summary())) #
#return(main$vegas_obj)
summary_res<-reticulate::py_to_r(main$vegas_obj$get_final_wt_results())
tolSuccess<-reticulate::py_to_r(main$vegas_obj$success)
summary_res<-as.list(c(summary_res,tolSuccess))
names(summary_res)<-c("mean","error","metTolerance")
return(summary_res)

# py$vegas$ravg(a$get_results())
#return(results)
#a$get_wt_result()

}

################################################################################
################################################################################
#' Bayesian Log Evidence via Vegas+
#'
#' @description tbc
#'
#' @details tbc
#'
#' @param f An R function that takes a matrix and returns a vector. See details and examples.
#' @param lower A vector of lower integration limits for each dimension, e.g. c(-1.,-1.-1)
#' @param upper A vector of upper integration limits for each dimension, e.g. c(1.,1.1)
#' @param nitn_warm Number of iterations for Vegas warmup
#' @param neval_warm Number of function evaluations per iteration in warmup
#' @param nitn Number of iterations post-warmup.
#' @param neval Number of function evaluations per iteration post-warmup.
#' @param errTol  the % error target, default is 1, i.e. error is 1% of current estimated integral value
#' @param maxIter max number of iteration blocks to run to achieve errTol. Each block comprises nitn iterations
#' @param seed random number seed for vegas sampling generating. set for reproducible results.
#' @param nsearch number of points to evaluate log_posterior to find approx max value for shiftby. See details.
#' @param extra_args a named list of additional arguments passed to the function f.
#' @export
vegasBayesEvidence <- function(f, lower,upper, nitn_warm = 10, neval_warm = 1000,
                  nitn = 10, neval = 1000, errTol=1,maxIter=5,seed=99999,
                  nsearch=1000,
                  extra_args=list()){

  set.seed(seed)
  start<-lower
  stop<-upper
  searchPts <- mapply(seq, from = start, to = stop,
                      MoreArgs = list(length.out = nsearch))
  for(i in 1:length(start)){searchPts[,i]<-searchPts[sample(1:nrow(searchPts)),i]}

  #print(extra_args)
  #print(searchPts)
  if(!"uselog"%in%names(extra_args)){stop("extra_args list must have named member 'uselog'. see help page")}
  if(!"shiftby"%in%names(extra_args)){stop("extra_args list must have named member 'shiftby'. see help page")}

  extra_args$uselog[1]<-1.0 # force this - use log return values
  extra_args$shiftby[1]<-0.0 # force this - no shifting, might get some -inf which is fine

  ## join search points into args as first argument and evaluation all search point
  mymax<-max(do.call(f,c(list(searchPts),extra_args)))

  # location shift - approx max over the integration domain
  cat("max log value found = ",mymax,"\n")

  extra_args$uselog[1]<-0.0 #force this
  extra_args$shiftby[1]<-mymax

  result<-vegas(f=f, lower=lower,upper=upper, nitn_warm = nitn_warm, neval_warm = neval_warm,
                nitn = nitn, neval = neval, errTol=errTol,maxIter=maxIter,
                seed = seed,
                extra_args = extra_args)

  log_evidence = mymax + log(result$mean)
  cat("log evidence = ",log_evidence,"\n")
  # should be around -129.4
  return(log_evidence)

}

