library(vegasr)
vegas_initialize()
library(mvtnorm)
mu<-matrix(c(0.5, -0.2, 0.1),nrow=1)
cov<-matrix(data=c(
                 1.0, 0.5, 0.2,
                 0.5, 1.2, 0.3,
                 0.2, 0.3, 0.8),ncol=3,byrow=FALSE)

myf<-function(x,mu,cov){
  res<-dmvnorm(x,mean = mu,sigma=cov)
  return(res)
  }

########### Estimate integal using defaults
########### cleanstart erases any previous results and grid
result<-vegas_integrate(f=myf,
                        lower=c(-5,-5,-5), upper=c(5,5,5),
                        nitn_warm = 10, neval_warm = 10000,
                        nitn = 10, neval = 10000,
                        errTol=0.01,maxIter=100,
                        mu=mu,cov=cov)
print(result)
#cat("Estimate = ",result$mean," error = ",result$error,"\n")


