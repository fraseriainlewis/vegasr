Sys.setenv('_R_CHECK_SYSTEM_CLOCK_' = 0)
devtools::load_all()
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
                        errTol=0.1,maxIter=20,
                        mu=mu,cov=cov)
print(result)
#cat("Estimate = ",result$mean," error = ",result$error,"\n")

########### Estimate 5-D integral with known value as check

alpha<-matrix(data=c(1.),nrow=1)
erf<-function(x){return(2 * pnorm(x * sqrt(2)) - 1)}
oneD<-sqrt(pi / alpha[1]) * erf(sqrt(alpha[1]) / 2.0)
true.val<-oneD**5
print(true.val)

gaus<-function(x,alpha){
  # x shape is (batch, dim)
  # Compute sum of squared differences from 0.5
  dx2 = apply((x-0.5)**2,1,sum)
  return(exp(-alpha[1] * dx2))
}
myx<-matrix(data=rep(0.7,10),ncol=5) # 2 row 5 col
gaus(x=myx,alpha=alpha)

result2<-vegas_integrate(f=gaus,
                        lower=rep(0,5), upper=rep(1,5),
                        nitn_warm = 10, neval_warm = 10000,
                        nitn = 5, neval = 10000,
                        errTol=0.01,maxIter=100,
                        alpha=alpha)
print(result2)










