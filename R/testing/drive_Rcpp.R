library(RcppArmadillo)
## provides dmvnorm_aram(x,my,cov)
Rcpp::sourceCpp("src/testing/test2.cpp")
#check_parallel()
library(vegasr)

theta   <- matrix(data=rep(0.1,length=4*6), ncol = 6)

arma_fn_log_post_1(theta, thedata$y, thedata$treat,0.0, 1.0)

vegasr:::fn_log_post_1(theta, thedata$y, thedata$treat,.0, 1.0)

library(vegasr)
# now setup python environment
vegas_initialize() # this needed called once per session after library(vegas)
library(tictoc)


tic()
result_logEv<-vegasBayesEvidence(f=vegasr:::fn_log_post_1,
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
result_logEv<-vegasBayesEvidence(f=arma_fn_log_post_1,
                                 lower=c(-1,-1,-1,-1,0.0001,0.0001),
                                 upper=c(1,1,1,1,1,1),
                                 nitn_warm = 5, neval_warm = 1e5,
                                 nitn = 5, neval = 1e5,
                                 errTol=1,maxIter=10,seed=99999,nsearch=10000,
                                 extra_args=list(
                                   y=thedata$y,treat=thedata$treat,shiftby=0,uselog=1.))
toc()
cat("log evidence = ",result_logEv,"\n")










# assume getwd() at root of package
library(mvtnorm)
library(RcppArmadillo)
## provides dmvnorm_aram(x,my,cov)
Rcpp::sourceCpp("R/testing/test1.cpp")

myf<-function(x,mu,cov,a=1.0){
  res<-dmvnorm(x,mean = mu,sigma=cov)*a[1]
  return(res)
}
# Example
x   <- matrix(rnorm(3*4), ncol = 3)
#mu  <- rep(0, 3)
#cov <- diag(3)

mu<-matrix(c(0.5, -0.2, 0.1),nrow=1)
cov<-matrix(data=c(
  1.0, 0.5, 0.2,
  0.5, 1.2, 0.3,
  0.2, 0.3, 0.8),ncol=3,byrow=FALSE)


dmvnorm_arma(x, mu, cov,2.0)

result<-vegas(f=myf,
                        lower=c(-5,-5,-5), upper=c(5,5,5),
                        nitn_warm = 10, neval_warm = 10000,
                        nitn = 10, neval = 10000,
                        errTol=0.1,maxIter=20,extra_args = list(
                        mu=mu,cov=cov,a=2.))
print(result)

result<-vegas(f=dmvnorm_arma,
                        lower=c(-5,-5,-5), upper=c(5,5,5),
                        nitn_warm = 10, neval_warm = 10000,
                        nitn = 10, neval = 10000,
                        errTol=0.1,maxIter=20,
              extra_args = list(
                          mu=mu,cov=cov,a=10.))
print(result)


myf2<-function(x,mu,cov){

  return(dmvnorm_arma(x, mu, cov))
}

#res<-dmvnorm_arma(x, mu, cov)
res<-dmvnorm(x, mu, cov)
res2<-myf(x,mu,cov)
print(res)
print(res2)

library(vegasr)
vegas_initialize()
result<-vegas_integrate(f=myf,
                        lower=c(-5,-5,-5), upper=c(5,5,5),
                        nitn_warm = 10, neval_warm = 10000,
                        nitn = 10, neval = 10000,
                        errTol=0.1,maxIter=20,
                        mu=mu,cov=cov)
print(result)
#cat("Estimate = ",result$mean," error = ",result$error,"\n")

result<-vegas_integrate(f=dmvnorm_arma,
                        lower=c(-5,-5,-5), upper=c(5,5,5),
                        nitn_warm = 10, neval_warm = 10000,
                        nitn = 10, neval = 10000,
                        errTol=0.1,maxIter=20,
                        mu=mu,cov=cov)
print(result)
#cat("Estimate = ",result$mean," error = ",result$error,"\n")


#######################################################################
f=dmvnorm_arma
lower=c(-5,-5,-5); upper=c(5,5,5);
nitn_warm = 10; neval_warm = 1000;
nitn = 10; neval = 1000;
errTol=0.01;maxIter=20;
mu=mu;cov=cov;
noms<-c("mu","cov")
vegasr_pyassign("mu",mu)
vegasr_pyassign("cov",cov)

