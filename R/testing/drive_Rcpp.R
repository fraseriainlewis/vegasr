# assume getwd() at root of package
library(mvtnorm)

myf<-function(x,mu,cov){
  res<-dmvnorm(x,mean = mu,sigma=cov)
  return(res)
}
## provides dmvnorm_aram(x,my,cov)
Rcpp::sourceCpp("R/testing/test1.cpp")

# Example
x   <- matrix(rnorm(3*4), ncol = 3)
mu  <- rep(0, 3)
cov <- diag(3)

res<-dmvnorm_arma(x, mu, cov)
res2<-myf(x,mu,cov)
print(res)
print(res2)
