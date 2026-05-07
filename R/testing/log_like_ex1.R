### setup data set
rm(list=ls())
library(extraDistr)
set.seed(99999)
# Set up data
rr_k_ctrl <- c(0.20)        # control response rate for each basket
rr_k_trt <- c(0.40)         # treatment response rate for each basket

K<-length(rr_k_ctrl)        # number of baskets

N_k_ctrl <- rep(100, K)     # number of control participants per basket
N_k_trt <- rep(100, K)      # number of treatment participants per basket
N_k <- N_k_ctrl + N_k_trt   # number of participants per basket (both arms combined)
N <- sum(N_k)               # total sample size
k_vec <- rep(1:K, N_k)      # N x 1 vector of basket indicators (1 to K)

z_vec<-NULL;
y<-NULL;
for(i in 1:K){ # for each basket repeat 0-control 1-trt according to the specifc Ns
  z_vec<-c(z_vec,rep(0:1,c(N_k_ctrl[i],N_k_trt[i]))) # treatment/control indicator
  y<-c(y,
       c(rbinom(N_k_ctrl[i],1,rr_k_ctrl[i]), # bernoulli for control
         rbinom(N_k_trt[i],1,rr_k_trt[i]))) #           for trt
}

thedata<-data.frame(y,basketID=k_vec,Treatment=z_vec)


# dhnorm(x, sigma, log = TRUE)
# dnorm(x,sigma, log=TRUE)

# pmin(pmax(x, -0.9999), 0.9999)

myx<-matrix(data=c(-0.11,-0.13,0.15,0.11,0.051,0.052,
                      -0.12,-0.1,0.17,0.12,0.052,0.051,
                      -0.13,-0.11,0.11,0.19,0.053,0.054
                      ),ncol=6,byrow=TRUE)
theta<-myx
#treat<-thedata$Treatment
#y<-thedata$y

y=thedata$y;treat=thedata$Treatment;scaling=matrix(data=rep(0.0,nrow(myx),ncol=1));s=0.

r_compute_log_lik<-function(theta, y,treat,scaling,s){

  theta0=pmin(pmax(theta[,1], -0.9999), 0.9999)
  theta1=pmin(pmax(theta[,2], -0.9999), 0.9999)
  theta2=pmin(pmax(theta[,3], -0.9999), 0.9999)
  theta3=pmin(pmax(theta[,4], -0.9999), 0.9999)
  theta4=pmin(pmax(theta[,5], 0.0001), 0.9999)
  theta5=pmin(pmax(theta[,6], 0.0001), 0.9999)

  jacobianL = (
    log1p(theta0^2) - 2.0*log1p(-(theta0^2))
    + log1p(theta1^2) - 2.0*log1p(-(theta1^2))
    + log1p(theta2^2) - 2.0*log1p(-(theta2^2))
    + log1p(theta3^2) - 2.0*log1p(-(theta3^2))
    + log1p(theta4^2) - 2.0*log1p(-(theta4^2))
    + log1p(theta5^2) - 2.0*log1p(-(theta5^2))

  )

  a0=theta0/(1-theta0^2)
  a1=theta1/(1-theta1^2)
  mu0=theta2/(1-theta2^2)
  mu1=theta3/(1-theta3^2)
  sigma0=theta4/(1-theta4^2)
  sigma1=theta5/(1-theta5^2)

  #eta <- treat %*% rbind(a1)   # (10,1) %*% (1,3) → (10,3)
  eta <- sweep(treat %*% rbind(a1), 2, a0, "+")

  #eta=a0+a1*treat # (10,3) where T_vec = (3,) and (10,1) broadcast to (10,3)
  # y_data*eta - mx.log(1+mx.exp(eta)) this is (10,3) - want col sums, so collapse over rows

  #logL = mx.sum(y*eta - mx.log1p(mx.exp(eta)),axis=0)
  #result <- sweep(B, 1, A, "*")

  #logL = apply(y*eta - log1p(exp(eta)),2,sum) #
  logL = apply(sweep(eta,1,y,"*") - log1p(exp(eta)),2,sum)

  #print(f"logL={logL}\n")
  # now add the priors
  # simple N(0,sd=10)
  prior_a0 = dnorm(a0,mean=mu0,sd=sigma0,log=TRUE)
  prior_a1 = dnorm(a1,mean=mu1,sd=sigma1,log=TRUE)
  prior_mu0 = dnorm(mu0,mean=0.,sd=2.5)
  prior_mu1 = dnorm(mu1,mean=0.,sd=2.5)
  prior_sigma0 = dhnorm(sigma0, sigma=2.5)
  prior_sigma1 = dhnorm(sigma1, sigma=2.5)

  logDens = logL + prior_a0 + prior_a1 + prior_mu0 + prior_mu1 + prior_sigma0 + prior_sigma1
  logPost = logDens + jacobianL

  if(s==1.){ # search phase
  return(logPost - scaling[1])
  } else return(exp(logPost - scaling[1]) )

    }

#mx.exp(log_lik - self.l_max)

r_compute_log_lik(theta=myx,y=thedata$y,treat=thedata$Treatment,scaling=0,s=1.)

set.seed(11111)
start<-c(-1,-1,-1,-1,0.0001,0.0001)
stop<-c(1,1,1,1,1,1)
searchPts <- mapply(seq, from = start, to = stop, MoreArgs = list(length.out = 10000))
for(i in 1:6){searchPts[,i]<-searchPts[sample(1:nrow(searchPts)),i]}

mymax<-max(r_compute_log_lik(theta=searchPts,y=thedata$y,treat=thedata$Treatment,
                             scaling=0.,
                             s=1.))
print(mymax)


y<-matrix(data=as.numeric(thedata$y),ncol=1)
treat<-matrix(data=as.numeric(thedata$Treatment),ncol=1)
#scaling<-matrix(data=rep(mymax,nrow(searchPts)),ncol=1)

#r_compute_log_lik(theta=myx,y=y,treat=treat,scaling=mymax,s=0.)

library(vegasr)
vegas_initialize()

result<-vegas_integrate(f=r_compute_log_lik,
                        lower=c(-1,-1,-1,-1,0.0001,0.0001), upper=c(1,1,1,1,1,1),
                        nitn_warm = 10, neval_warm = 10000,
                        nitn = 10, neval = 10000,
                        errTol=0.1,maxIter=10,
                        y=y,treat=treat,scaling=mymax,s=0.)
print(result)
#cat("Estimate = ",result$mean," error = ",result$error,"\n")
log_evidence = mymax + log(result$mean)
cat("log evidence = ",log_evidence," result$mean = ",result$mean,"\n")







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










