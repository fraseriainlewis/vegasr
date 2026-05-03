#' Multidimensional Integration using Vegas
#'
#' @param f An R function that takes a numeric vector and returns a numeric value.
#' @param lower A vector of lower integration limits for each dimension, e.g. c(-1.,-1.-1)
#' @param upper A vector of upper integration limits for each dimension, e.g. c(1.,1.1)
#' @param nitn_warm Number of iterations for warmup
#' @param neval_warm Number of function evaluations per iteration in warmup
#' @param nitn Number of iterations.
#' @param neval Number of function evaluations per iteration.
#' @param cleanstart Logical, default TRUE means erase any stored results and include grid warmup. See details.
#' @param ... Additional arguments passed to the vegas integrator.
#' @importFrom glue glue
#' @examples
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
#' ########### Estimate integal using defaults
#' ########### cleanstart erases any previous results and grid
#' result<-vegas_integrate(f=myf,
#'                        lower=c(-5,-5,-5), upper=c(5,5,5),
#'                        nitn_warm = 10, neval_warm = 1000,
#'                        nitn = 5, neval = 1000,
#'                        cleanstart=TRUE,
#'                        mu=mu,cov=cov)
#' cat("Estimate = ",result$mean," error = ",result$error,"\n")
#'
#' ########## Add additional iterations to improve precision
#' result<-vegas_integrate(f=myf,
#'                         lower=c(-5,-5,-5),upper=c(5,5,5),
#'                         nitn_warm = 10, neval_warm = 1000,
#'                         nitn = 10, neval = 1000,
#'                         cleanstart=FALSE,
#'                         mu=mu,cov=cov)
#' cat("New Estimate = ",result$mean," new error = ",result$error,"\n")
#' @export
vegas_integrate <- function(f, lower,upper, nitn_warm = 10, neval_warm = 1000,
                                            nitn = 10, neval = 1000, cleanstart=TRUE,
                                            ...) {

  if (is.null(getOption("vegas_initialized"))) {
    vegas_initialize()
  }

  run_checks() # empty - to be completed - named argument checks etc

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

  main$cleanstart<-as.logical((cleanstart[1]))

  main$nitn_warm<-as.integer((nitn_warm[1]))
  main$neval_warm<-as.integer((neval_warm[1]))
  main$nitn<-as.integer((nitn[1]))
  main$neval<-as.integer((neval[1]))

  if (length(list(...)) > 0){
    #cat("parsing additional arguments\n")
    m<-length(list(...));
    args<-list(...)
    noms<-names(args)
    for(i in 1:m){
      print(args[[i]])
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

# Initialize the integrator
if cleanstart:
    vegas_obj.clear_results()
    integ2 = vegas.Integrator([[l, u] for l, u in zip(lower, upper)])
    # Adaptation phase # no results stored
    integ2(newf, nitn=nitn_warm, neval=neval_warm)

# integration storing interative results, still potentially adapting grid
result2 = integ2(newf, nitn=nitn, neval=neval)
vegas_obj.add_results(result2) # save into object
#print(result2.summary())
',.trim=FALSE)

bigstring<-paste(stringpart,sep="")
#  return(bigstring)
reticulate::py_run_string(bigstring)
main <- reticulate::import_main(convert = FALSE)
cat(reticulate::py_to_r(main$result2$summary()))
#return(main$vegas_obj)
summary_res<-reticulate::py_to_r(main$vegas_obj$get_final_wt_results())
names(summary_res)<-c("mean","error")
summary_res<-as.list(summary_res)
return(summary_res)

# py$vegas$ravg(a$get_results())
#return(results)
#a$get_wt_result()

}


