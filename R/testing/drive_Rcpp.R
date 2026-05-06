# assume getwd() at root of package
library(mvtnorm)

## provides dmvnorm_aram(x,my,cov)
Rcpp::sourceCpp("R/testing/test1.cpp")

myf<-function(x,mu,cov){
  res<-dmvnorm(x,mean = mu,sigma=cov)
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

