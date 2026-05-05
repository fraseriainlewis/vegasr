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
#' @param maxIter max number of iteration blocks to run to achieve errTol. Each block comprises nitn iterations.
#' @param ... Additional arguments passed to the function f. These must be numeric vectors or matrices. See details.
#' @importFrom glue glue
#' @examples
#' \dontrun{
#' library(vegasr)
#' vegas_initialize() # only needed once per session
#'
#' ### EXAMPLE 1 - use of additional matrix arguments
#' library(mvtnorm)
#' mu<-matrix(c(0.5, -0.2, 0.1),nrow=1)
#' cov<-matrix(data=c(
#'                 1.0, 0.5, 0.2,
#'                 0.5, 1.2, 0.3,
#'                 0.2, 0.3, 0.8),ncol=3,byrow=FALSE)
#'
#'myf<-function(x,mu,cov){
#'  res<-dmvnorm(x,mean = mu,sigma=cov)
#'  return(res)
#'  }
#'
#' result<-vegas_integrate(f=myf,
#'                         lower=c(-5,-5,-5), upper=c(5,5,5),
#'                         nitn_warm = 10, neval_warm = 10000,
#'                         nitn = 10, neval = 10000,
#'                         errTol=0.1,maxIter=20,
#'                         mu=mu,cov=cov)
#' print(result)
#' # mean should be very close to 0.999 and error close to 0.005
#'
#' ########### Estimate 5-D integral with known solution to check correctness
#' # define integrand
#' gaus<-function(x,alpha){
#'        # x shape is (batch, dim)
#'        # alpha is scalar - note the alpha[1] below
#'        # Compute sum of squared differences from 0.5
#'        dx2 = apply((x-0.5)**2,1,sum)
#'        return(exp(-alpha[1] * dx2))
#'  }
#'  # check function will work with matrix argument and return vector
#'  myx<-matrix(data=rep(0.7,10),ncol=5) # 2 row 5 col
#'  alpha<-matrix(data=c(1.),nrow=1) # scalar stored in matrix
#'  print(gaus(x=myx,alpha=alpha)) # should return vector of length 2
#'
#'  # compute Vegas estimate
#'  result2<-vegas_integrate(f=gaus,lower=rep(0,5), upper=rep(1,5),
#'  nitn_warm = 10, neval_warm = 10000, nitn = 5, neval = 10000,
#'  errTol=0.01,maxIter=100,
#'  alpha=alpha)
#'  print(result2)
#'
#'  #compute known numerical result
#'  alpha<-matrix(data=c(1.),nrow=1)
#'  erf<-function(x){return(2 * pnorm(x * sqrt(2)) - 1)}
#'  oneD<-sqrt(pi / alpha[1]) * erf(sqrt(alpha[1]) / 2.0)
#'  true.val<-oneD**5
#'  print(true.val)
#'  cat("Vegas =",result2$mean," known estimate =",true.val,"\n")
#'
#' }
#'
#'
#'
#' @export
vegas_integrate <- function(f, lower,upper, nitn_warm = 10, neval_warm = 1000,
                                            nitn = 10, neval = 1000, errTol=1,maxIter=5,
                                            ...) {

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

  if (length(list(...)) > 0){
    #cat("parsing additional arguments\n")
    m<-length(list(...));
    args<-list(...)
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
  return r_func({str2})

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

    #print(i)
    #print(ests)
    #print(result2.summary())
    i+=1

if ests[1]>((RerrTol/100)*ests[0]):
    vegas_obj.success=False

',.trim=FALSE)

bigstring<-paste(stringpart,sep="")
#  return(bigstring)
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


