library(RcppArmadillo)
library(RcppEigen)
library(RcppParallel)
library(tictoc)
## provides dmvnorm_aram(x,my,cov)

#Rcpp::sourceCpp("src/testing/eigen_v1.cpp")
Rcpp::sourceCpp("src/testing/fns_imp_eigen.cpp")

library(vegasr)
## Important - run next line in each R session to ensure python is ready
vegas_initialize()
myf<-function(x,mu,cov){
  res<-dmvnorm(x,
               mean = mu, # this is a 1-row matrix, dmvnorm accepts this
               sigma=cov)
  return(res)
}

mu<-matrix(c(0.5, -0.2, 0.1),nrow=1) # note matrix
cov<-matrix(data=c(
  1.0, 0.5, 0.2,
  0.5, 1.2, 0.3,
  0.2, 0.3, 0.8),ncol=3,byrow=FALSE)

## See help page for descriptions of warm and nitn and neval.
vegas_result<-vegasip(f=myf,
                      lower=c(-0.5,-0.5,-0.5), upper=c(2.,1.,3.),
                      nitn_warm = 10, neval_warm = 10000,
                      nitn = 10, neval = 10000,
                      errTol=0.1, maxIter=20,seed=99999,adapt=TRUE,
                      extra_args=list(mu=mu, cov=cov))
# extra_args are additional arguments needed for myf

###### COMPUTE GRID JACOBIAN and weights now check that the jac

