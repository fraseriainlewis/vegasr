library(RcppArmadillo)
library(RcppEigen)
library(RcppParallel)
library(tictoc)
## provides dmvnorm_aram(x,my,cov)
Rcpp::sourceCpp("src/testing/arma_v1.cpp")

Rcpp::sourceCpp("src/testing/eigen_v1.cpp")
Rcpp::sourceCpp("src/testing/eigen_v1_par.cpp")

Rcpp::sourceCpp("src/testing/rcmp_parallel_test.cpp")

#tic()
#myexp<-parallel_arma_exp(matrix(rnorm(1e7),ncol=1))
#toc()





library(vegasr)

thedata<-vegasr:::fn_create_data_1(99999)
theta   <- matrix(data=rep(0.1,length=4*6), ncol = 6)

arma_fn_log_post_1(theta, thedata$y, thedata$treat,0.0, 1.0)

eigen_fn_log_post_1(theta, thedata$y, thedata$treat,0.0, 1.0)

eigen_fn_log_post_1_par(theta, thedata$y, thedata$treat,0.0, 1.0)


library(vegasr)
# now setup python environment
vegas_initialize() # this needed called once per session after library(vegas)
library(tictoc)


tic()
result_logEv<-vegasBayesEvidence(f=arma_fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=0.1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
cat("log evidence = ",result_logEv,"\n")


tic()
result_logEv<-vegasBayesEvidence(f=eigen_fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=0.1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
cat("log evidence = ",result_logEv,"\n")

tic()
result_logEv<-vegasBayesEvidence(f=eigen_fn_log_post_1_par,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=0.1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
cat("log evidence = ",result_logEv,"\n")


library(vegasr)
vegas_initialize()
tic()
mymarg<-vegasBayesPosterior(f=vegasr:::fn_marg_1_1,
                            lower=c(-1,-1,-1,0.0001,0.0001),
                            upper=c(1,1,1,1,1),
                            nitn_warm = 10, neval_warm = 10000,
                            nitn = 10, neval = 10000,
                            errTol=1,maxIter=10,seed=99999,nsearch=10000,
                            log_evidence = result_logEv,
                            extra_args=list(
                              y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.,z=-1.))
toc()
cat("Marginal density f(z) at z = -1. = ",mymarg,"\n")

library(vegasr)
vegas_initialize()
tic()
mymarg<-vegasBayesPosterior(f=eigen_fn_marg_1_1_par,
                            lower=c(-1,-1,-1,0.0001,0.0001),
                            upper=c(1,1,1,1,1),
                            nitn_warm = 10, neval_warm = 10000,
                            nitn = 10, neval = 10000,
                            errTol=1,maxIter=10,seed=99999,nsearch=10000,
                            log_evidence = result_logEv,
                            extra_args=list(
                              y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.,z=-1.))
toc()
cat("Marginal density f(z) at z = -1. = ",mymarg,"\n")



